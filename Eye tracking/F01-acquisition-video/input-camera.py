import cv2

# 0 = webcam intégrée par défaut
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur : image non reçue")
        break

    cv2.imshow("Webcam", frame)

    # Appuyer sur q pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()