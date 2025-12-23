import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import cv2
import glob

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 50
LATENT_DIM_HINTS = "Autoencoder Convolucional Profundo"
THRESHOLD_K = 2.0  # Fator K para o limiar (μ + k*σ)

# Caminho base do dataset (AJUSTE AQUI PARA SUA PASTA LOCAL)
DATASET_PATH = './dataset' 

# ==========================================
# 1. PRÉ-PROCESSAMENTO E CARREGAMENTO
# ==========================================
def load_and_preprocess_image(path):
    """
    Lê imagem, converte p/ grayscale, resize e normaliza (0-1).
    """
    img = cv2.imread(path)
    if img is None: return None
    
    # Conversão para Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize fixo
    img = cv2.resize(img, IMG_SIZE)
    
    # Opcional: Equalização de Histograma (melhora contraste)
    img = cv2.equalizeHist(img)
    
    # Normalização (0-1) e Adição do canal de cor (H, W, 1)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    
    return img

def build_dataset(base_path, split, class_filter=None):
    """
    Carrega imagens de um split (train/val/test).
    Se class_filter for definido (ex: 'healthy'), carrega apenas essa classe.
    """
    images = []
    labels = [] # 0: Healthy, 1: Anomaly
    
    # Procura recursivamente
    search_path = os.path.join(base_path, split, '**', '*.jpeg') # Ajuste extensão se necessário (jpg/jpeg/dcm)
    file_paths = glob.glob(search_path, recursive=True)
    
    # Fallback para outras extensões
    if not file_paths:
        file_paths = glob.glob(os.path.join(base_path, split, '**', '*.jpeg'), recursive=True)

    print(f"[{split.upper()}] Encontradas {len(file_paths)} imagens.")

    for fpath in file_paths:
        folder_name = os.path.basename(os.path.dirname(fpath)).lower()
        
        # Lógica de Rótulos
        is_healthy = 'healthy' in folder_name or 'normal' in folder_name
        
        # Filtro para TREINO (Apenas saudáveis)
        if class_filter == 'healthy' and not is_healthy:
            continue
            
        img = load_and_preprocess_image(fpath)
        if img is not None:
            images.append(img)
            labels.append(0 if is_healthy else 1)
            
    return np.array(images), np.array(labels)

# Carregando Datasets
print("--- Carregando Dados ---")
# Treino: APENAS SAUDÁVEIS
x_train, _ = build_dataset(DATASET_PATH, 'train', class_filter='healthy')

# Validação: SAUDÁVEIS (para calcular threshold) e ANOMALIAS (para check visual)
x_val, y_val = build_dataset(DATASET_PATH, 'val')

# Teste: AMBOS
x_test, y_test = build_dataset(DATASET_PATH, 'test')

print(f"Shapes -> Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# ==========================================
# 2. ARQUITETURA DO MODELO (CAE)
# ==========================================
def build_autoencoder(input_shape):
    # --- Encoder ---
    input_img = layers.Input(shape=input_shape)
    
    # Camadas convolucionais para extrair textura e estrutura
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) # 128x128
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) # 64x64
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x) # 32x32 (Latent Space)

    # --- Decoder ---
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x) # 64x64
    
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # 128x128
    
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # 256x256
    
    # Saída Sigmoid para garantir intervalo [0, 1]
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    return autoencoder

autoencoder = build_autoencoder((IMG_SIZE[0], IMG_SIZE[1], 1))
autoencoder.compile(optimizer='adam', loss='mse') # MSE foca em erros estruturais
autoencoder.summary()

# ==========================================
# 3. TREINAMENTO
# ==========================================
# Early Stopping para evitar overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, mode='min'
)

# Separa apenas saudáveis da validação para monitorar a convergência correta
x_val_healthy = x_val[y_val == 0]

print("\n--- Iniciando Treinamento (Apenas Saudáveis) ---")
history = autoencoder.fit(
    x_train, x_train, # Entrada = Saída (Reconstrução)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_val_healthy, x_val_healthy),
    callbacks=[early_stopping],
    verbose=1
)

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss (Healthy)')
plt.legend()
plt.title('Curva de Aprendizado')
plt.show()

# ==========================================
# 4. DEFINIÇÃO DO LIMIAR (THRESHOLD)
# ==========================================
print("\n--- Calculando Limiar Estatístico ---")

# Reconstruir imagens de validação SAUDÁVEIS
reconstructions_healthy = autoencoder.predict(x_val_healthy)
# Calcular erro médio absoluto (MAE) por imagem
train_loss = tf.keras.losses.mae(reconstructions_healthy, x_val_healthy)
# O retorno é (N, H, W), precisamos da média por imagem
train_loss = np.mean(train_loss, axis=(1, 2))

# Estatísticas
mean_loss = np.mean(train_loss)
std_loss = np.std(train_loss)

# Definir Threshold
threshold = mean_loss + (THRESHOLD_K * std_loss)
print(f"Média do Erro (Saudável): {mean_loss:.5f}")
print(f"Desvio Padrão: {std_loss:.5f}")
print(f"THRESHOLD DEFINIDO: {threshold:.5f} (μ + {THRESHOLD_K}σ)")

# Plot da distribuição dos erros
plt.hist(train_loss, bins=50, alpha=0.7, label='Saudáveis (Val)')
plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
plt.title('Distribuição de Erro de Reconstrução (Saudáveis)')
plt.legend()
plt.show()

# ==========================================
# 5. AVALIAÇÃO E DETECÇÃO
# ==========================================
def predict_anomaly(model, images, threshold):
    reconstructions = model.predict(images)
    # Erro por imagem
    loss = tf.keras.losses.mae(reconstructions, images)
    loss = np.mean(loss, axis=(1, 2))
    
    # Classificação baseada no limiar
    predictions = (loss > threshold).astype(int)
    
    return predictions, loss, reconstructions

print("\n--- Avaliando no Conjunto de Teste ---")
preds, scores, recons = predict_anomaly(autoencoder, x_test, threshold)

# Métricas
print(classification_report(y_test, preds, target_names=['Normal', 'Anômalo']))

# Matriz de Confusão
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anômalo'], yticklabels=['Normal', 'Anômalo'])
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# ==========================================
# 6. VISUALIZAÇÃO DO MAPA DE ERRO
# ==========================================
def plot_anomaly_map(original, reconstructed, label, score, threshold):
    # Calcula diferença absoluta (mapa de erro)
    error_map = np.abs(original - reconstructed)
    # Remove dimensão do canal para plotar
    original = original.squeeze()
    reconstructed = reconstructed.squeeze()
    error_map = error_map.squeeze()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original (Label: {label})")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Reconstrução (IA)")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Mapa de Erro (Score: {score:.4f} / Th: {threshold:.4f})")
    # Usa 'jet' ou 'hot' para destacar onde o erro é alto
    plt.imshow(error_map, cmap='inferno') 
    plt.colorbar()
    plt.axis('off')
    
    plt.show()

print("\n--- Visualizando Exemplos de Teste ---")
# Visualizar 3 Anomalias e 3 Saudáveis do teste
indices_anomaly = np.where(y_test == 1)[0][:3]
indices_healthy = np.where(y_test == 0)[0][:3]

for idx in np.concatenate((indices_anomaly, indices_healthy)):
    plot_anomaly_map(x_test[idx], recons[idx], y_test[idx], scores[idx], threshold)