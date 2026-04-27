#This will contain the gui code that will display the preprocessed data from preprocess.py
import sys
import argparse
import numpy
import preprocess
import cv2
numpy.float = numpy.float64
numpy.int = numpy.int_

from PySide6 import QtCore, QtWidgets, QtGui

class QtDemo(QtWidgets.QWidget):
    def __init__(self, video_path, results, frame_count, events):
        super().__init__()

        self.button = QtWidgets.QPushButton("Next Frame")
        self.video_path = video_path
        self.frame_count = frame_count
        self.events = events
        self.results = results
        self.current_frame = 0

        self.stats_label = QtWidgets.QLabel(f"Total Hits: {len(self.events['hits'])}, Total Bounces: {len(self.events['bounces'])}")


        # Configure image label
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.render_frame(0)

        print("[INFO] Image shape: ", self.frame_count)

        # Configure slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frame_count-1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.stats_label)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.frame_slider)

        # Connect functions
        self.button.clicked.connect(self.on_click)
        self.frame_slider.sliderMoved.connect(self.on_move)

    @QtCore.Slot()
    def on_click(self):
        if self.current_frame == self.frame_count-1:
            return
        self.current_frame += 1
        self.frame_slider.setValue(self.current_frame)
        self.render_frame(self.current_frame)

    @QtCore.Slot()
    def on_move(self, pos):
        self.current_frame = pos
        self.render_frame(self.current_frame)

    def render_frame(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame ", idx)
            cap.release()
            return
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if idx in self.results["frame"]:
            result = self.results["frame"].index(idx)
            objs = self.results["objects"][result]
        else:
            objs = []
        
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(img)
        painter = QtGui.QPainter(pixmap)
        for obj in objs:
            if obj["bbox"] is None:
                continue
            x, y = int(obj["x"]), int(obj["y"])
            bbox = obj["bbox"]
            painter.setPen(QtGui.QPen(QtCore.Qt.red, 2))
            painter.drawRect(bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0])
            painter.setPen(QtGui.QPen(QtCore.Qt.green, 2))
            painter.drawEllipse(x-5, y-5, 10, 10)
            history = obj["history"]
            if len(history) > 1:
                painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
                for i in range(len(history)-1):
                    x1, y1 = int(history[i][0]), int(history[i][1])
                    x2, y2 = int(history[i+1][0]), int(history[i+1][1])
                    painter.drawLine(x1, y1, x2, y2)
        painter.end()
        
        self.img_label.setPixmap(pixmap)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo for loading video with Qt5.")
    parser.add_argument("video_path", metavar='PATH_TO_VIDEO', type=str)
    parser.add_argument("--num_frames", metavar='n', type=int, default=-1)
    parser.add_argument("--grey", metavar='True/False', type=str, default=False)
    args = parser.parse_args()


    results, events, frame_count = preprocess.preprocess(args.video_path, alpha=3, tau=60, delta=75, s=1, N=3, hit_threshold=12, bounce_threshold=6)
    print("Hits: ", len(events["hits"]), "at frames: ", events["hits"])
    print("Bounces: ", len(events["bounces"]), "at frames: ", events["bounces"])
    print("Total frames: ", frame_count)
    video_path = args.video_path

    app = QtWidgets.QApplication([])

    widget = QtDemo(video_path, results, frame_count, events)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
