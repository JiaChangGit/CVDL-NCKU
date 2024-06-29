import sys
import os
import re
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import question1, question2, question3, question4, question5


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        loadUi("./qtUI.ui", self)
        self.loadAllFile = ""
        self.files = []
        self.leftFig = ""
        self.rightFig = ""
        self.wide = 11
        self.height = 8
        self.obj_point = np.zeros((self.wide * self.height, 3), dtype=np.float32)
        self.obj_point[:, :2] = np.mgrid[0 : self.wide, 0 : self.height].T.reshape(
            -1, 2
        )
        self.matrix00 = None
        self.matrix01 = None
        # Q4 member
        self.loadQ4_filename1 = ""
        self.loadQ4_filename2 = ""
        # Q5 member
        self.load_filenameForQ5 = ""

        self.Connect_btn()

    def Connect_btn(self):
        self.pushButton_1.clicked.connect(self.pushButton1F)
        self.pushButton_2.clicked.connect(self.pushButton2F)
        self.pushButton_3.clicked.connect(self.pushButton3F)
        self.pushButton_4.clicked.connect(self.pushButton4F)
        self.pushButton_5.clicked.connect(self.pushButton5F)
        self.find_extrinsic_btn.clicked.connect(self.find_extrinsic_btnF)
        self.pushButton_7.clicked.connect(self.pushButton7F)
        self.pushButton_8.clicked.connect(self.pushButton8F)
        self.pushButton_9.clicked.connect(self.pushButton9F)
        self.pushButton_10.clicked.connect(self.pushButton10F)
        self.pushButton_11.clicked.connect(self.pushButton11F)
        self.pushButton_12.clicked.connect(self.pushButton12F)
        self.pushButton_13.clicked.connect(self.pushButton13F)
        self.pushButton_14.clicked.connect(self.pushButton14F)
        self.pushButton_15.clicked.connect(self.pushButton15F)
        self.pushButton_16.clicked.connect(self.pushButton16F)
        self.pushButton_17.clicked.connect(self.pushButton17F)
        self.pushButton_18.clicked.connect(self.pushButton18F)
        self.pushButton_19.clicked.connect(self.pushButton19F)
        self.pushButton_20.clicked.connect(self.pushButton20F)

    def pushButton1F(self):
        self.loadAllFile = str(
            QFileDialog.getExistingDirectory(self, "Select Directory")
        )
        print(self.loadAllFile)
        overall = [i for i in os.listdir(self.loadAllFile) if re.search(".bmp", i)]
        self.files = sorted(overall, key=lambda x: int(x.split(".")[0]))
        print(self.files)
        # f_name = file_name[0]
        # df = pd.DataFrame(pd.read_gbq(f_name))

    def pushButton2F(self):
        self.leftFig = str(QFileDialog.getOpenFileName(self, "open file", ".")[0])
        imgLeft = cv2.imread(self.leftFig)
        print(imgLeft)

    def pushButton3F(self):
        self.rightFig = str(QFileDialog.getOpenFileName(self, "open file", ".")[0])
        imgRight = cv2.imread(self.rightFig)
        print(imgRight)

    def pushButton4F(self):
        question1.findCorners(self)

    def pushButton5F(self):
        question1.findInstrinsic(self)

    def find_extrinsic_btnF(self):
        question1.findExtrinsic(self, self.find_extrinsic_combobox.currentText())

    def pushButton7F(self):
        question1.findDistorsion(self)

    def pushButton8F(self):
        question1.showResultClick(self)

    ################ Q2
    def pushButton9F(self):
        question2.horizontallyClick(self, self.lineEdit.text())

    def pushButton10F(self):
        question2.verticallyClick(self, self.lineEdit.text())

    ################ Q3
    def pushButton11F(self):
        question3.disparityMap(self)

    ############### Q4
    def pushButton12F(self):
        self.loadQ4_filename1 = str(
            QFileDialog.getOpenFileName(self, "Choose a file")[0]
        )
        print(self.loadQ4_filename1)

    def pushButton13F(self):
        self.loadQ4_filename2 = str(
            QFileDialog.getOpenFileName(self, "Choose a file")[0]
        )
        print(self.loadQ4_filename2)

    def pushButton14F(self):
        question4.createKeyPoint(self)

    def pushButton15F(self):
        question4.matchedKeyPoint(self)

    ############### Q5
    def pushButton16F(self):
        question5.loadClick(self)

    def pushButton17F(self):
        question5.augmentClick(self)

    def pushButton18F(self):
        question5.structClick()

    def pushButton19F(self):
        question5.accClick(self)

    def pushButton20F(self):
        question5.inferClick(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
