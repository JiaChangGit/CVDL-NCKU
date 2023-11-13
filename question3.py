import cv2


def disparityMap(self):
    if self.leftFig and self.rightFig:
        imgLeft = cv2.imread(self.leftFig)
        imgRight = cv2.imread(self.rightFig)
        imgL_gray = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM_create(numDisparities=21 * 16, blockSize=19)
        disparity = stereo.compute(imgL_gray, imgR_gray)
        disparity = cv2.normalize(
            disparity,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        disparity_show = cv2.cvtColor(disparity, cv2.COLOR_GRAY2RGB)
        cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("disparity", 700, 600)
        cv2.imshow("disparity", disparity_show)

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                imgR_cp = imgRight.copy()
                imgL_cp = imgLeft.copy()
                if disparity[y, x] != 0:
                    cv2.circle(
                        imgR_cp,
                        (x - disparity[y, x] - 50, y),
                        24,
                        (255, 255, 0),
                        thickness=-1,
                    )
                    print("disparity: {}".format(disparity[y, x]))
                    print(
                        "depth: {}".format(
                            int(342.789 * 4019.284 / (279.184 + disparity[y, x]))
                        )
                    )
                    cv2.imshow("imgRight", imgR_cp)
                    cv2.imshow("imgLeft", imgL_cp)

        cv2.namedWindow("imgLeft", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("imgLeft", 800, 700)
        cv2.namedWindow("imgRight", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("imgRight", 800, 700)
        cv2.imshow("imgLeft", imgLeft)
        cv2.imshow("imgRight", imgRight)
        cv2.setMouseCallback("imgLeft", mouse)
        cv2.waitKey()

    cv2.destroyAllWindows()
