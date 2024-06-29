import cv2
import numpy as np
import os


def findCorners(self):
    # termination criterias
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (self.wide, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            img_out = cv2.drawChessboardCorners(
                img, (self.wide, self.height), new_corners, ret
            )
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
            cv2.namedWindow("Corners", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Corners", 1024, 1024)
            cv2.imshow("Corners", img_out)
            cv2.waitKey(1000)
    self.matrix00 = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    cv2.destroyAllWindows()


def findInstrinsic(self):
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.wide, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
    self.matrix00 = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    print("Intrinsic:", self.matrix00[1], sep="\n")


def findExtrinsic(self, combox):
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.wide, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
    self.matrix00 = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    i = int(combox) - 1
    rotation_mat = cv2.Rodrigues(self.matrix00[3][i])[0]
    extrinsic_mat = np.hstack([rotation_mat, self.matrix00[4][i]])
    print("Extrinsic", extrinsic_mat, sep="\n")


def findDistorsion(self):
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.wide, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
    self.matrix00 = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    print("Distorsion:", self.matrix00[2], sep="\n")


def showResultClick(self):
    criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.wide, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
    self.matrix00 = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        h, w = img.shape[:2]
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(
            self.matrix00[1], self.matrix00[2], (w, h), 0, (w, h)
        )
        dst = cv2.undistort(
            img, self.matrix00[1], self.matrix00[2], None, newcameramatrix
        )

        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]
        img = cv2.resize(img, (480, 480))
        dst = cv2.resize(dst, (480, 480))
        imgs = np.hstack([dst, img])
        cv2.imshow("undistorted result", imgs)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
