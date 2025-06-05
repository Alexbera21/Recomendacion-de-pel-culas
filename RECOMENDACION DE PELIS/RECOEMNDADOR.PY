import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# FUNCIONES CLAVE
# ------------------------------

def cargar_imagenes(directorio):
    imagenes = []
    nombres = []
    for archivo in os.listdir(directorio):
        if archivo.lower().endswith(("posters/cartel-jurassic-world-1-carlos-villarin-freelance.jpg", "posters/download.jpg","posters/vengadores-infinity-war-anthony-y-joe-russo-2018.webp")):
            path = os.path.join(directorio, archivo)
            imagen = cv2.imread(path)
            if imagen is not None:
                imagenes.append(imagen)
                nombres.append(archivo)
    return imagenes, nombres

def extraer_histograma(imagen, bins=(8, 8, 8)):
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([imagen_rgb], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def similitud_coseno_manual(v1, v2):
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom if denom != 0 else 0

def obtener_recomendaciones(index, features, nombres, top_k=5):
    base_vector = features[index]
    similitudes = []
    for i, feature in enumerate(features):
        if i != index:
            score = similitud_coseno_manual(base_vector, feature)
            similitudes.append((nombres[i], score))
    similitudes.sort(key=lambda x: x[1], reverse=True)
    return similitudes[:top_k]

def mostrar_resultados(pelicula_base, recomendaciones, directorio):
    fig, axs = plt.subplots(1, len(recomendaciones)+1, figsize=(15, 5))
    
    img_base = cv2.imread(os.path.join(directorio, pelicula_base))
    img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    axs[0].imshow(img_base)
    axs[0].set_title("Película Base")
    axs[0].axis('off')
    
    for i, (nombre, score) in enumerate(recomendaciones):
        img = cv2.imread(os.path.join(directorio, nombre))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i+1].imshow(img)
        axs[i+1].set_title(f"{nombre}\nSimilitud: {score:.2f}")
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

# ------------------------------
# PROGRAMA PRINCIPAL
# ------------------------------

def recomendador_visual(directorio_posters, indice_pelicula=0):
    imagenes, nombres = cargar_imagenes(directorio_posters)
    if not imagenes:
        print("No se encontraron imágenes en el directorio.")
        return

    features = [extraer_histograma(img) for img in imagenes]
    
    recomendaciones = obtener_recomendaciones(indice_pelicula, features, nombres)
    
    mostrar_resultados(nombres[indice_pelicula], recomendaciones, directorio_posters)

# ------------------------------
# USO DEL SISTEMA
# ------------------------------

# Ruta al directorio que contiene los posters
directorio_posters = 'posters'  # Asegúrate de tener este folder con imágenes

# Recomendaciones para la película en la posición 0
recomendador_visual(directorio_posters, indice_pelicula=0)
