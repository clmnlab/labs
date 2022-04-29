

# Common Libraries
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import time

# Custom Libraries

# Sources

class Watcom(QLabel):
    def __init__(self,
                 parent_rect,
                 width,
                 height,
                 is_drawing = True,
                 sampling_move_interval = 0):
        """
        :param parent_rect: parent rect
        :param width: width
        :param height: height
        :param is_drawing: is use drawing
        :param sampling_move_interval: second
        """
        super().__init__()

        # variable to check watcom window is loaded
        self.is_loaded = False

        # Window size
        parent_x, parent_y = parent_rect.x(), parent_rect.y()
        parent_width, parent_height = parent_rect.width(), parent_rect.height()

        # Move to Center
        x_pos = parent_x + int(parent_width / 2) - int(width / 2)
        y_pos = parent_y + int(parent_height / 2) - int(height / 2)
        self.move(x_pos, y_pos)

        p = self.palette()
        # p.setColor(self.backgroundRole(), Qt.white)
        # self.setAutoFillBackground(True)
        self.setWindowOpacity(0.6) # 투명도
        self.setPalette(p)

        self.myPixmap = QPixmap(width,height)
        self.resize(width, height)
        self.painter = QPainter(self.myPixmap)
        self.pen = QPen(Qt.black)
        self.painter.setPen(self.pen)
        self.painter.fillRect(0, 0, width, height, Qt.white)
        self.setPixmap(self.myPixmap)

        self.is_drawing = is_drawing
        self.sampling_move_interval = sampling_move_interval

        self.last_move_record_time = None
        self.event_proc = None

    def tabletEvent(self, tabletEvent):
        self.pen_pos = tabletEvent.pos()

        self.pen_pressure = int(tabletEvent.pressure() * 100)

        if tabletEvent.type() == QTabletEvent.TabletPress:
            self.pen_last = self.pen_pos
            self.last_move_record_time = time.time()

            self.event_occurred(QTabletEvent.TabletPress, self.pen_pos.x(), self.pen_pos.y(), self.pen_pressure)
        elif tabletEvent.type() == QTabletEvent.TabletMove:
            if self.is_drawing == True:
                # draw line
                if self.pen_last != None and self.pen_pos != None:
                    self.painter.drawLine(self.pen_last, self.pen_pos)
                self.setPixmap(self.myPixmap)

                self.pen_last = self.pen_pos

                self.update()

            if self.last_move_record_time != None:
                current_time = time.time()
                if current_time - self.last_move_record_time > self.sampling_move_interval:
                    self.last_move_record_time = current_time

                    self.event_occurred(QTabletEvent.TabletMove, self.pen_pos.x(), self.pen_pos.y(), self.pen_pressure)

        elif tabletEvent.type() == QTabletEvent.TabletRelease:
            self.pen_last = None
            self.last_move_record_time = None

            self.event_occurred(QTabletEvent.TabletRelease, self.pen_pos.x(), self.pen_pos.y(), self.pen_pressure)

        tabletEvent.accept()

    def updateSize(self, width, height):
        pm = QPixmap(width, height)
        pm.fill(Qt.white)
        old = self.myPixmap
        self.myPixmap = pm
        self.pen = QPen(Qt.black)
        self.painter = QPainter(pm)
        self.painter.drawPixmap(0,0,old)
        self.setPixmap(pm)

    def resizeEvent(self, event):
        if event.oldSize().width() > 0:
            self.updateSize(event.size().width(), event.size().height())

    def event_occurred(self, event_type, x, y, pen_pressure):
        if self.event_proc != None:
            self.event_proc(self, event_type, x, y, pen_pressure)

    def set_event_interface(self, func):
        """
        set event interface

        :param func: callback function
        """
        self.event_proc = func

    def set_is_drawing(self, is_drawing):
        self.is_drawing = is_drawing

    def set_opacity(self, opacity):
        self.setWindowOpacity(opacity)

    def show_withCallback(self, load_callback):
        """
        show window with call callback function

        :param load_callback: function(is_loaded)
        """
        self.show()
        self.is_loaded = True

        load_callback()

def open(rect,
         event_proc, # arg: event_type, x, y
         is_drawing = True,
         sampling_move_interval = 0.5):
    parent_rect = QRect(rect)

    app = QApplication(sys.argv)

    win = Watcom(parent_rect=parent_rect,
                 width=parent_rect.width(),
                 height=parent_rect.height(),
                 is_drawing=is_drawing,
                 sampling_move_interval = sampling_move_interval)

    win.set_event_interface(event_proc)
    win.setWindowFlag(Qt.WindowType.FramelessWindowHint)
    win.show()
    app.exec_()

if __name__=="__main__":
    app = QApplication(sys.argv)
    parent_rect = app.desktop().frameGeometry()

    win = Watcom(parent_rect=parent_rect,
                 width=parent_rect.width(),
                 height=parent_rect.height(),
                 is_drawing=True,
                 sampling_move_interval=2)

    def event_proc(win, event_type, x, y, pressure):
        print(win, event_type, x, y, pressure)

    win.set_event_interface(event_proc)
    win.setWindowFlag(Qt.WindowType.FramelessWindowHint)
    win.show()

    app.exec_()