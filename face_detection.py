import sys
import cv2
import os
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from encode import encode_known_faces
from validate import validate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Face Detection System')
        self.setGeometry(100, 80, 800, 600)

        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Setup GUI components
        self.image_label = QLabel(self)
        self.take_sample_button = QPushButton('Take Sample', self)
        self.train_data_button = QPushButton('Train Data', self)
        self.validate_button = QPushButton('Validate', self)

        self.take_sample_button.clicked.connect(self.take_sample)
        self.train_data_button.clicked.connect(self.train_data)
        self.validate_button.clicked.connect(self.validate_photo)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.take_sample_button)
        layout.addWidget(self.train_data_button)
        layout.addWidget(self.validate_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def take_sample(self):
        folder_name, ok = QInputDialog.getText(self, 'Input', 'Enter folder name:')
        if ok and folder_name:
            folder_path = Path("training") / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            num_photos = 5
            photo_count = 0

            while photo_count < num_photos:
                ret, frame = self.video_capture.read()
                if not ret:
                    logging.error("Failed to grab frame")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    face_image = frame[y:y+h, x:x+w]
                    photo_path = folder_path / f"photo_{photo_count + 1}.jpg"
                    cv2.imwrite(str(photo_path), face_image)
                    photo_count += 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Display captured frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_rgb.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(q_image))
                

            logging.info(f"Captured {num_photos} photos in folder {folder_name}")

    def train_data(self):
        encode_known_faces()
        logging.info("Training complete.")

    def validate_photo(self):
        ret, frame = self.video_capture.read()
        if not ret:
            logging.error("Failed to grab frame for validation.")
            return

        photo_path = "temp_validation.jpg"
        cv2.imwrite(photo_path, frame)
        validate(photo_path)
        os.remove(photo_path)
        logging.info("Validation complete.")

    def closeEvent(self, event):
        self.video_capture.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = FaceDetectionApp()
    main_win.show()
    sys.exit(app.exec_())
