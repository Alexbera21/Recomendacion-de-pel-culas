https://github.com/giankxlos/TRABAJO-FINALLLLLLLLLXciones)+1, figsize=(15, 5))
    
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
