#Ecuaseñas 
#Desarrollado por Patricia Constante
#***************************************************
# Guarda en un archivo CSV las coordenadas normalizadas o landmarkes de la detección de una mano,  dando una etiqueta al usuario.
# permite sobreescribir o no el archivo CSV
# Para la creación de base de datos de imagen y además guarda la imagen con la detección de los 21 keypoints
# Se crea un código para identificar al usuario del que se obtiene la imagen y además la seña que realiza. 

import cv2
import mediapipe as mp
import csv
import os

# Inicializa los módulos de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

user=1

# Archivo CSV para guardar datos
output_file = "hand_landmarks_label.csv"
image_folder = "saved_images"  # Carpeta para guardar las imágenes
image_folder2 = "saved_images2"  # Carpeta para guardar las imágenes procesadas

# Crear carpeta si no existe
os.makedirs(image_folder, exist_ok=True)
os.makedirs(image_folder2, exist_ok=True)

# Crear encabezado reordenado
header = []
for i in range(21):  # 21 landmarks
    header.extend([f"x{i}", f"y{i}", f"z{i}"])
header.append("label")  # Agregar columna para la etiqueta
header.append("username") # Agregar columna para la username


# Preguntar al usuario si desea sobrescribir o continuar escribiendo en el archivo
file_mode = "a"
if os.path.exists(output_file):
    choice = input(f"El archivo '{output_file}' ya existe. ¿Deseas sobrescribirlo? (s/n): ").lower()
    if choice == 's':
        file_mode = "w"

# Crear archivo CSV si se elige sobrescribir o no existe
if file_mode == "w" or not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Escribir encabezado

# Captura de video
cap = cv2.VideoCapture(0)
print("Presiona 's' para guardar landmarks con etiqueta, 'u' para cambiar / asignar código de usuario, o 'Esc' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar manos
    results = hands.process(frame_rgb)

    # Dibujar landmarks si se detecta una mano
    frame_keypoints = frame.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibuja los landmarks en la imagen
            mp.solutions.drawing_utils.draw_landmarks(frame_keypoints, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar la imagen procesada
    cv2.imshow("MediaPipe Hands", frame_keypoints)

    # Capturar eventos de teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u'):  # Si presionas 'u', ingresa el nombre
            user = input("Ingresa tu nombre de usuario: ")
            print(f"Nombre de usuario ingresado: {user}")
    if key == ord('s'):  # Presiona 's' para guardar landmarks
        if results.multi_hand_landmarks:
            # Capturar landmarks de la primera mano detectada
            hand_landmarks = results.multi_hand_landmarks[0]
            data_row = []

            # Extraer coordenadas x, y, z de los 21 landmarks en orden
            for landmark in hand_landmarks.landmark:
                data_row.extend([landmark.x, landmark.y, landmark.z])

            # Pedir al usuario que ingrese un label
            label = input("Ingresa la etiqueta para estos landmarks: ")
            data_row.append(label)

             # Agregar label, nombre de imagen y usuario a los datos
            user_code = f"u{str(user).zfill(2)}"  # Asignar dos cifras al nombre de usuario
            label_code = f"s{str(label).zfill(2)}"  # Asignar dos cifras a la etiqueta
    
            username= f"{user_code}{label_code}"
            
            data_row.append(username)
            #data_row.append(image_name)
          

            # Guardar en el archivo CSV
            with open(output_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(data_row)

             # Crear el nombre de la imagen usando el formato user_label
            image_name = f"{username}.png"
            keypoints_image_name = f"{username}_kp.png"

            # Guardar imágenes en sus respectivas carpetas
            image_path = os.path.join(image_folder, image_name)
            keypoints_image_path = os.path.join(image_folder2, keypoints_image_name)

            cv2.imwrite(image_path, frame)  # Imagen original
            cv2.imwrite(keypoints_image_path, frame_keypoints)  # Imagen con puntos clave
            

            print(f"Datos guardados con etiqueta '{username}'.")

    elif key == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()
print(f"Datos guardados en {output_file}.")

