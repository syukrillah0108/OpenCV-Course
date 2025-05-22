import cv2
import numpy as np

# URL stream
stream_url = 'http://localhost:8080/?action=stream'

# Buka stream video
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Gagal membuka stream!")
    exit()

# Muat pre-trained model untuk deteksi objek (misalnya MobileNet SSD)
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt', 
    'mobilenet_iter_73000.caffemodel'
)

# Label objek yang terdeteksi (misalnya 20 label untuk dataset COCO)
class_names = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor'
]

while True:
    # Baca frame dari stream
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame!")
        break

    # Ubah ukuran frame agar lebih cepat diproses
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    # Berikan input ke model
    net.setInput(blob)
    detections = net.forward()

    # Iterasi melalui hasil deteksi
    detected_objects = []  # List untuk menyimpan objek yang terdeteksi
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Ambang batas confidence
            # Ambil koordinat kotak pembatas dan label objek
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Dapatkan label objek
            label = class_names[int(detections[0, 0, i, 1])]
            detected_objects.append(label)  # Simpan objek yang terdeteksi

            # Gambar kotak pembatas dan label (dihapus untuk tidak menampilkan preview)

    # Cetak objek yang terdeteksi pada setiap frame
    if detected_objects:
        print("Objek terdeteksi: ", ", ".join(detected_objects))

    # Tekan 'q' untuk keluar (tidak digunakan karena tidak ada preview)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Tutup stream
cap.release()
