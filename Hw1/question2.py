import cv2
import numpy as np
import os


def calibaration2(self):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    point_3D = []
    point_2D = []
    for file in self.files:
        img = cv2.imread(os.path.join(self.loadAllFile, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (self.wide, self.height), None)
        if ret == True:
            new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)
            point_3D.append(self.obj_point)
            point_2D.append(new_corners)
    # ret, intrinsic, distort, r_vecs, t_vecs
    self.matrix01 = cv2.calibrateCamera(
        point_3D, point_2D, gray.shape[::-1], None, None
    )


def draw(img, corners, img_point, len):
    img_point = np.int32(img_point).reshape(-1, 2)
    for i in range(len):
        img = cv2.line(
            img, tuple(img_point[2 * i]), tuple(img_point[2 * i + 1]), (0, 0, 255), 15
        )
    return img


def horizontallyClick(self, inWord):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibaration2(self)

    word = []
    text = inWord.upper()
    lib = os.path.join(self.loadAllFile, "Q2_lib/alphabet_lib_onboard.txt")
    fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)

    length = 0
    for i in range(len(text)):
        if text[i].encode("UTF-8").isalpha() and not text[i].isdigit():
            word.append(fs.getNode(text[i]).mat())
            length = length + 1

    pos_adjust = [[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]]

    for i in range(length):
        for j in range(len(word[i])):
            new_axis1 = [a + b for a, b in zip(word[i][j][0], pos_adjust[i])]
            new_axis2 = [a + b for a, b in zip(word[i][j][1], pos_adjust[i])]
            word[i][j][0] = new_axis1
            word[i][j][1] = new_axis2

    for i in range(len(self.files)):
        img = cv2.imread(os.path.join(self.loadAllFile, self.files[i]))
        rotation_vector = cv2.Rodrigues(self.matrix01[3][i])[0]
        transform_vector = self.matrix01[4][i]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.wide, self.height), None)
        new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

        # ret, intrinsic, distort, r_vecs, t_vecs
        axis = []
        for j in range(len(word)):
            axis1 = np.array(word[j], dtype=np.float32).reshape(-1, 3)
            axis.append(axis1)
            img_points, jac = cv2.projectPoints(
                axis[j],
                rotation_vector,
                transform_vector,
                self.matrix01[1],
                self.matrix01[2],
            )
            img = draw(img, new_corners, img_points, len(word[j]))

        cv2.namedWindow("Augmented Reality", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Augmented Reality", 480, 480)
        cv2.imshow("Augmented Reality", img)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()


def verticallyClick(self, inWord):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibaration2(self)

    word = []
    text = inWord.upper()
    lib = os.path.join(self.loadAllFile, "Q2_lib/alphabet_lib_vertical.txt")
    fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)

    length = 0
    for i in range(len(text)):
        if text[i].encode("UTF-8").isalpha() and not text[i].isdigit():
            word.append(fs.getNode(text[i]).mat())
            length = length + 1

    pos_adjust = [[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]]

    for i in range(length):
        for j in range(len(word[i])):
            new_axis1 = [a + b for a, b in zip(word[i][j][0], pos_adjust[i])]
            new_axis2 = [a + b for a, b in zip(word[i][j][1], pos_adjust[i])]
            word[i][j][0] = new_axis1
            word[i][j][1] = new_axis2

    for i in range(len(self.files)):
        img = cv2.imread(os.path.join(self.loadAllFile, self.files[i]))
        rotation_vector = cv2.Rodrigues(self.matrix01[3][i])[0]
        transform_vector = self.matrix01[4][i]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.wide, self.height), None)
        new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

        axis = []
        for j in range(len(word)):
            axis1 = np.array(word[j], dtype=np.float32).reshape(-1, 3)
            axis.append(axis1)
            img_points, jac = cv2.projectPoints(
                axis[j],
                rotation_vector,
                transform_vector,
                self.matrix01[1],
                self.matrix01[2],
            )
            img = draw(img, new_corners, img_points, len(word[j]))

        cv2.namedWindow("Augmented Reality_V", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Augmented Reality_V", 480, 480)
        cv2.imshow("Augmented Reality_V", img)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
