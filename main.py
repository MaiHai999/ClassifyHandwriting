
from PyQt5.QtWidgets import *
import cv2
from tensorflow import keras
import sys
import string



from interface.DrawingCanvas import DrawingCanvas

class HandwritingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recogntion Handwright App")
        self.model = keras.models.load_model("preTrain/my_model.h5")

        self.uppercase_alphabet = list(string.ascii_uppercase)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.drawingCanvas = DrawingCanvas()
        layout.addWidget(self.drawingCanvas)

        group_button_box = QWidget()
        button_layout = QHBoxLayout()
        group_button_box.setLayout(button_layout)

        classify_button = QPushButton("Classify", self)
        classify_button.clicked.connect(self.classify_character)
        button_layout.addWidget(classify_button)

        clear_button = QPushButton("Clear", self)
        clear_button.clicked.connect(self.drawingCanvas.clear_drawing)
        button_layout.addWidget(clear_button)

        layout.addWidget(group_button_box)

        self.lable_result = QLabel()
        layout.addWidget(self.lable_result)

    def classify_character(self):
        self.drawingCanvas.save_as_jpg("picture.png")

        img = cv2.imread('picture.png' , 0)
        image = cv2.resize(img, (28,28), interpolation =  cv2.INTER_AREA)
        _, thresholded_image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
        inverted_image = ~thresholded_image
        input_img = inverted_image.flatten().reshape(1, -1)

        predictions = self.model.predict(input_img)
        result = self.uppercase_alphabet[int(predictions.argmax(axis=1))]
        text = "Character is " + result + " ,accuracy: " + str(predictions.max(axis=1)[0])

        self.lable_result.setText(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HandwritingApp()
    window.show()
    sys.exit(app.exec_())


