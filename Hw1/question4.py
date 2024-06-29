import cv2
import numpy as np


def createKeyPoint(self):
    if self.loadQ4_filename1 == "":
        print("no data load")
        return
    img1 = cv2.imread(self.loadQ4_filename1)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    SIFT = cv2.SIFT_create()
    key1, des1 = SIFT.detectAndCompute(img1_gray, None)

    kp_image1 = cv2.drawKeypoints(img1_gray, key1, np.array([]), color=(0, 255, 0))
    cv2.imshow("Keypoints", kp_image1)

    cv2.waitKey()
    cv2.destroyAllWindows()


def matchedKeyPoint(self):
    if self.loadQ4_filename1 == "" or self.loadQ4_filename2 == "":
        return

    img1 = cv2.imread(self.loadQ4_filename1)
    img2 = cv2.imread(self.loadQ4_filename2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    SIFT = cv2.SIFT_create()
    key1, des1 = SIFT.detectAndCompute(img1_gray, None)
    key2, des2 = SIFT.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)

    goodMatches = []
    minRatio = 0.75
    for m, n in matches:
        if m.distance < minRatio * n.distance:
            goodMatches.append(m)

    if len(goodMatches) > 10:
        src_pts = np.float32([key1[m.queryIdx].pt for m in goodMatches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([key2[m.trainIdx].pt for m in goodMatches]).reshape(
            -1, 1, 2
        )

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
        matchesMask = mask.ravel().tolist()

    draw_params = dict(
        matchColor=(0, 255, 255),
        singlePointColor=(0, 255, 0),
        matchesMask=matchesMask,
        flags=2,
    )

    # img1 = cv2.drawKeypoints(img1, key1, np.array([]), color=(0, 255, 0))
    img3 = cv2.drawMatches(img1, key1, img2, key2, goodMatches, None, **draw_params)
    cv2.imshow("Matched", img3)
    cv2.waitKey()
    cv2.destroyAllWindows()
