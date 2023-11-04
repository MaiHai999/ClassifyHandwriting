
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class DrawingCanvas(QLabel):
    def __init__(self):
        super().__init__()
        pixmap = QPixmap(300, 200)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#000000')

    def set_pen_color(self, c):
        self.pen_color = QColor(c)

    def mouseMoveEvent(self, e):
        if self.last_x is None:  # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        painter = QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(4)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def save_as_jpg(self, filename):
        # Chụp nội dung của QLabel đã vẽ thành một QPixmap
        pixmap = self.pixmap()

        # Lưu pixmap thành một tệp hình ảnh JPEG
        pixmap.save(filename, "png")

    def clear_drawing(self):
        pixmap = QPixmap(self.width(), self.height())  # Create a new pixmap with the same size as the canvas
        pixmap.fill(Qt.white)  # Fill it with a white background
        self.setPixmap(pixmap)  # Set the pixmap as the new canvas

        # Reset the last_x and last_y values
        self.last_x, self.last_y = None, None


