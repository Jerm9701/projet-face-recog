import cv2

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    # Touche espace : capture
    if key == 32:
        filename = f"F02-calibration-camera\\calib_imgs\\calib_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image enregistrée : {filename}")
        count += 1

    # Touche q : quitter
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()