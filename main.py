from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore
import sys
import numpy as np
import cv2
import torch
from torchsummary import summary
from torchvision import models
from infer import infer_by_test_idx
import torch.nn as nn
from random_erasing import showcase

class Ui(object):
    def __init__(self) -> None:
        super().__init__()
        self.MainWindow =  QtWidgets.QMainWindow()
    def setupUi(self):
        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(312, 312)
        
        self.centralwidget = QtWidgets.QWidget(self.MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.button = QtWidgets.QPushButton(self.centralwidget)
        self.button.setGeometry(QtCore.QRect(50, 20, 220, 31))
        self.button.setObjectName("button")
        
        self.button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.button_2.setGeometry(QtCore.QRect(50, 60, 220, 32))
        self.button_2.setObjectName("button_2")
        
        self.button_3 = QtWidgets.QPushButton(self.centralwidget)
        self.button_3.setGeometry(QtCore.QRect(50, 100, 220, 32))
        self.button_3.setObjectName("button_3")
        
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(50, 140, 220, 32))
        self.lineEdit.setObjectName("lineEdit")


        self.button_4 = QtWidgets.QPushButton(self.centralwidget)
        self.button_4.setGeometry(QtCore.QRect(50, 180, 220, 22))
        self.button_4.setObjectName("button_4")
        
        self.button_5 = QtWidgets.QPushButton(self.centralwidget)
        self.button_5.setGeometry(QtCore.QRect(50, 220, 220, 22))
        self.button_5.setObjectName("button_5")

        self.MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(self.MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 252, 32))
        self.menubar.setObjectName("menubar")
        self.MainWindow.setMenuBar(self.menubar)
                

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)
    

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.MainWindow.setWindowTitle(_translate("MainWindow", "Question 5"))
        self.button.setText("Show Model Structure")
        self.button_2.setText("Show TensorBord")
        self.button_3.setText("Test")
        self.button_4.setText("Data Augumentation ShowCase")
        self.button_5.setText("Comparison Example")
        

        self.button.clicked.connect(show_summary)
        self.button_2.clicked.connect(show_tensorBord)
        self.button_3.clicked.connect(lambda: infer(int(self.lineEdit.text())))
        self.button_4.clicked.connect(re_showcase)
        self.button_5.clicked.connect(compare_models_acc)
    
class My_window(QMainWindow, Ui):
    def __init__(self, parent=None):
        super(My_window, self).__init__(parent)
        self.setupUi(self)
        self.initUI()
        self.index = 1
        self.img = None


model = models.resnet50()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load('./models/resNet50.model', map_location=torch.device('cpu')))

def show_summary():
    summary(model, (3, 32, 32))        

def infer(idx):
    pred, img_path = infer_by_test_idx(idx)
    
    img = cv2.imread(img_path)
    cv2.imshow(pred, img)
    close()

def show_tensorBord():
    img = cv2.imread("./img/ScreenShot.png")    
    
    cv2.imshow("Load Image", img)
    close()

def re_showcase():    
    img = showcase()
    img = np.array(img)
    cv2.imshow("Show Case", img)
    close()

def compare_models_acc():
    img = cv2.imread("./img/output.png")
    cv2.imshow("Load Image", img)
    close()


def close():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)    

    ui = Ui()
    ui.setupUi()

    ui.MainWindow.show()
    
    sys.exit(app.exec_())
