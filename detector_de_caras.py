import cv2

# Cargar el clasificador de cascada Haar para la detección de caras
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()
    
    # Convertir la imagen a escala de grises para la detección de caras
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar las caras en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Dibujar un rectángulo alrededor de las caras detectadas
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    # Mostrar la imagen con las caras detectadas
    cv2.imshow('frame',frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()