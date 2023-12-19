import os
import sys
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsPixmapItem,
)
from PyQt5.uic import loadUi
import cvdl2_Q1toQ3


class VGG19_BN(nn.Module):
    def __init__(self):
        super(VGG19_BN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


transform_without_erasing = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)


def loadModel_withoutE():
    model_without_erasing = models.resnet50()
    model_without_erasing.fc = nn.Linear(model_without_erasing.fc.in_features, 2)
    model_without_erasing.load_state_dict(torch.load("model_without_erasing.pth"))
    return model_without_erasing


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        loadUi("./cvdl2.ui", self)
        self.setGraph()

        self.video = ""
        self.images = ""
        self.folder = ""

        self.Connect_btn()

    def Connect_btn(self):
        self.pushButton_1.clicked.connect(self.loadImageClick)
        self.pushButton_2.clicked.connect(self.loadVideoClick)
        self.pushButton_3.clicked.connect(self.pushButton3F)
        self.pushButton_4.clicked.connect(self.pushButton4F)
        self.pushButton_5.clicked.connect(self.pushButton5F)
        self.pushButton_6.clicked.connect(self.pushButton6F)
        self.pushButton_7.clicked.connect(self.show_model_Q4)
        self.pushButton_8.clicked.connect(self.showAccuracy_Q4)
        self.pushButton_9.clicked.connect(self.predict_Q4)
        self.pushButton_10.clicked.connect(self.reset_Q4)
        self.pushButton_11.clicked.connect(self.loadClick)
        self.pushButton_12.clicked.connect(self.showImages)
        self.pushButton_13.clicked.connect(self.modelClick)
        self.pushButton_14.clicked.connect(self.comparisonClick)
        self.pushButton_15.clicked.connect(self.inferenceClick)

    def setGraph(self):
        img = QImage(520, 240, QImage.Format_RGB32)
        img.fill(QColor(0, 0, 0))
        pixmap = QPixmap.fromImage(img)
        self.label_1.setText("")
        self.label_1.setPixmap(pixmap)
        self.label_1.mousePressEvent = self.mousePress
        self.label_1.mouseMoveEvent = self.mouseMove

    def mousePress(self, event):
        # print("Press")
        img = QPixmap(self.label_1.pixmap())
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(5)
        painter = QPainter(img)
        painter.setPen(pen)
        painter.drawLine(
            event.pos().x(), event.pos().y(), event.pos().x(), event.pos().y()
        )
        painter.end()
        self.label_1.setPixmap(img)

    def mouseMove(self, event):
        # print("Move")
        img = QPixmap(self.label_1.pixmap())
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(25)
        painter = QPainter(img)
        painter.setPen(pen)
        painter.drawLine(
            event.pos().x(), event.pos().y(), event.pos().x(), event.pos().y()
        )
        painter.end()
        self.label_1.setPixmap(img)

    def loadImageClick(self):
        self.images = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
        print(self.images)

    def loadVideoClick(self):
        self.video = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
        print(self.video)

    def pushButton3F(self):
        cvdl2_Q1toQ3.subtractionClick(self)

    def pushButton4F(self):
        cvdl2_Q1toQ3.preprocessingClick(self)

    def pushButton5F(self):
        cvdl2_Q1toQ3.trackingClick(self)

    def pushButton6F(self):
        cvdl2_Q1toQ3.dimensionReductionClick(self)

    ### Q4 ###
    def show_model_Q4(self):
        model = VGG19_BN()
        summary(model, input_size=(1, 32, 32))

    def showAccuracy_Q4(self):
        file = "training_validation_plot.png"
        self.images = np.array(Image.open(file), dtype=np.uint8)
        self.images = cv2.cvtColor(self.images, cv2.COLOR_BGR2RGB)
        # cv2.imshow("image", self.images)
        # cv2.waitKey(0)

        frame = QImage(file)
        pix = QPixmap.fromImage(frame)
        pix = pix.scaled(520, 240)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.label_1.setPixmap(pix)

    def predict_Q4(self):
        preprocess = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ]
        )
        img = self.label_1.pixmap().toImage()
        # QImage
        img.save("q4_test.jpg")
        img = Image.open("q4_test.jpg")
        img = preprocess(img)
        img = img.unsqueeze(0)
        print("Trans Complete")
        net = VGG19_BN().to("cpu")
        checkpoint = torch.load("cvdl2_Q4_vgg19_bn_model.pth", map_location="cpu")
        net.load_state_dict(checkpoint, strict=False)
        print("Load Model")
        net.eval()

        # Predict Picture
        with torch.no_grad():
            output = net(img)
        print(output)

        Output = list(output[0])
        for i in range(len(Output)):
            if Output[i] < 0:
                Output[i] = 0
        labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        plt.bar(labels, Output)
        plt.show()

    def reset_Q4(self):
        # print("Reset")
        img = QImage(520, 240, QImage.Format_RGB32)
        img.fill(QColor(0, 0, 0))
        pixmap = QPixmap.fromImage(img)
        self.label_1.setText("")
        self.label_1.setPixmap(pixmap)

    ### Q5 ###
    def loadClick(self):
        self.images = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
        print(self.images)

    def showImages(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        _cat = os.listdir(os.path.join(path, "Cat"))[0]
        _dog = os.listdir(os.path.join(path, "Dog"))[0]
        cat = cv2.imread(os.path.join(path, "Cat", _cat))
        dog = cv2.imread(os.path.join(path, "Dog", _dog))

        cat = cv2.resize(cat, (224, 224), interpolation=cv2.INTER_AREA)
        dog = cv2.resize(dog, (224, 224), interpolation=cv2.INTER_AREA)

        cat = cv2.cvtColor(cat, cv2.COLOR_BGR2RGB)
        dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Cat")
        plt.imshow(cat)
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Dog")
        plt.imshow(dog)
        plt.show()

    def modelClick(self):
        model = loadModel_withoutE()
        print("\nModel without Random Erasing:")
        print(model)
        summary(model)

    def comparisonClick(self):
        accuracy = cv2.imread("Comparison.png")
        cv2.imshow("", accuracy)

    def inferenceClick(self):
        if self.images == "":
            self.loadClick()
        # Load and preprocess the image
        model = loadModel_withoutE()
        image = Image.open(self.images).convert("RGB")
        image = transform_without_erasing(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            output = F.softmax(model(image), dim=1)
            print(output)
        print(output[0][0].item())

        # Determine the predicted class label
        threshold = 0.24  # wrong with threshold = 0.5
        if output[0][0].item() < threshold:
            predicted_class = "Cat"
        else:
            predicted_class = "Dog"

        # Display the result
        self.label_3.setText(f"Predict: {predicted_class}")

        # Display the result image
        result_pixmap = QPixmap(self.images)
        self.label_2.setPixmap(result_pixmap)
        self.label_2.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
