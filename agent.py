import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
import matplotlib.pyplot as plt

CAMINHO_BASE = "images"
TAMANHO_IMAGEM = (64, 64)

def carregar_imagem(caminho, tamanho=TAMANHO_IMAGEM):
    try:
        img = Image.open(caminho).convert("L")  # tons de cinza (raio-X)
        img = img.resize(tamanho)
        return np.asarray(img).flatten() / 255.0
    except:
        return None

print("Carregando imagens...")
X = []
y = []

for classe, rotulo in [("normal", 0), ("pneumonia", 1)]:
    pasta = os.path.join(CAMINHO_BASE, classe)
    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith((".jpg", ".png", ".jpeg")):
            caminho_img = os.path.join(pasta, arquivo)
            vetor = carregar_imagem(caminho_img)
            if vetor is not None:
                X.append(vetor)
                y.append(rotulo)

X = np.array(X)
y = np.array(y)

print(f"{X.shape[0]} imagens carregadas")

modelo = SVC(kernel='rbf', gamma='scale', class_weight='balanced')
modelo.fit(X, y)

print("Modelo treinado com sucesso!")

IMG_TESTE = "teste.jpeg"

img_teste = carregar_imagem(IMG_TESTE)

if img_teste is not None:
    resultado = modelo.predict([img_teste])

    print("\nResultado da análise:")
    if resultado[0] == 1:
        print("Este paciente TEM pneumonia.")
    else:
        print("Este paciente NÃO tem pneumonia.")

    # Exibe a imagem
    img_visual = Image.open(IMG_TESTE).convert("L")
    plt.imshow(img_visual, cmap="gray")
    plt.title("Raio-X analisado")
    plt.axis("off")
    plt.show()
else:
    print("Erro ao carregar imagem de teste.")
