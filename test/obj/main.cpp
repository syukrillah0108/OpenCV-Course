#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Memuat model Haar Cascade untuk wajah
    string cascadePath = "haarcascade_frontalface_default.xml"; // Path ke file xml Haar Cascade
    CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) {
        cerr << "Error loading cascade file!" << endl;
        return -1;
    }

    // Inisialisasi stream video (dari localhost)
    VideoCapture cap("http://localhost:8080/?action=stream");  // Ganti dengan URL stream video Anda
    if (!cap.isOpened()) {
        cerr << "Gagal membuka stream video!" << endl;
        return -1;
    }

    Mat frame;
    while (cap.read(frame)) {
        // Konversi gambar ke grayscale (Haar Cascade bekerja di grayscale)
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // Deteksi wajah
        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(30, 30));

        // Gambarkan kotak di sekitar wajah yang terdeteksi
        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2);
        }

        // Tampilkan hasil deteksi di terminal
        cout << "Jumlah wajah terdeteksi: " << faces.size() << endl;

        // Menunggu 66 ms untuk mencapai 15 fps
        if (waitKey(66) >= 0) break; // Menunggu 66 ms (15 fps)
    }

    return 0;
}
