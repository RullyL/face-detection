import cv2

face_ref = cv2.CascadeClassifier('face_ref.xml')

# Mengaktifkan kamera (0 = kamera utama)
camera = cv2.VideoCapture(0)


def face_detection(frame):
    optimeze_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah di gambar dengan classifier
    faces = face_ref.detectMultiScale(
        optimeze_frame,
        scaleFactor=1.1,       # besar gambar diperkecil untuk pencarian wajah
        minNeighbors=5         # semakin besar nilainya, semakin ketat deteksinya
    )
    return faces


def drawer_box(frame):
    # Untuk setiap wajah yang terdeteksi, gambar kotak merah di sekitarnya
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)


def close_win():
    camera.release()
    cv2.destroyAllWindows()
    exit()


def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        # Tampilkan frame dengan nama jendela 'Face_ai'
        cv2.imshow('Face_ai', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_win()


if __name__ == "__main__":
    main()
