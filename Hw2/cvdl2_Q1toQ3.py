import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA


def subtractionClick(self):
    videoCap = cv2.VideoCapture(self.video)
    ret, frame = videoCap.read()
    if not ret:
        return

    # Create background subtractor
    history = 500
    distThreshold = 400
    subtractor = cv2.createBackgroundSubtractorKNN(
        history, distThreshold, detectShadows=True
    )

    while True:
        # Read frame
        ret, frame = videoCap.read()
        if not ret:
            break

        # Blur frame
        blurredFrame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Get background mask
        masks = subtractor.apply(blurredFrame)

        # Generate Frame (R) with only moving object by cv2.bitwise_and
        result = cv2.bitwise_and(frame, frame, mask=masks)

        # Show the frame with the background mask
        cv2.imshow("frame", frame)
        cv2.imshow("mask", masks)
        cv2.imshow("result", result)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    videoCap.release()
    cv2.destroyAllWindows()


def preprocessingClick(self):
    videoCap = cv2.VideoCapture(self.video)
    ret, frame = videoCap.read()
    if not ret:
        return

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adjust parameters for goodFeaturesToTrack
    maxCorners = 1
    qualityLevel = 0.3
    minDistance = 7
    blockSize = 7

    # Detect corners using goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners, qualityLevel, minDistance, blockSize
    )

    if corners is not None:
        # Get the coordinates of the corner
        x, y = corners[0][0]

        # Draw a red cross mark at the corner point, set the length of the line to 20 pixels, and the line thickness to 4 pixels
        cv2.line(frame, (int(x) - 10, int(y)), (int(x) + 10, int(y)), (0, 0, 255), 4)
        cv2.line(frame, (int(x), int(y) - 10), (int(x), int(y) + 10), (0, 0, 255), 4)

    # Show the frame with the cross mark
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 960, 540)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)

    videoCap.release()
    cv2.destroyAllWindows()


def trackingClick(self):
    videoCap = cv2.VideoCapture(self.video)
    ret, prev_frame = videoCap.read()
    if not ret:
        return

    # Convert previous frame to grayscale
    prevGray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Adjust parameters for goodFeaturesToTrack
    maxCorners = 1
    qualityLevel = 0.3
    minDistance = 7
    blockSize = 7

    # Detect corners using goodFeaturesToTrack
    prevCorners = cv2.goodFeaturesToTrack(
        prevGray, maxCorners, qualityLevel, minDistance, blockSize
    )

    # Create an empty mask image
    mask = np.zeros_like(prev_frame)

    while True:
        ret, frame = videoCap.read()
        if not ret:
            break

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using cv2.calcOpticalFlowPyrLK
        nextCorners, status, _ = cv2.calcOpticalFlowPyrLK(
            prevGray, gray, prevCorners, None
        )

        # Select good points
        goodNew = nextCorners[status == 1]
        goodOld = prevCorners[status == 1]

        # Draw trajectory lines
        for i, (new, old) in enumerate(zip(goodNew, goodOld)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 100, 255), 2)
            frame = cv2.line(
                frame, (int(a) - 10, int(b)), (int(a) + 10, int(b)), (0, 0, 255), 4
            )
            frame = cv2.line(
                frame, (int(a), int(b) - 10), (int(a), int(b) + 10), (0, 0, 255), 4
            )
            # frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 100, 255), -1)

        # Overlay trajectory lines on the frame
        output = cv2.add(frame, mask)

        # Show the frame with the trajectory lines
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 960, 540)
        cv2.imshow("frame", output)

        if cv2.waitKey(15) & 0xFF == ord("q"):
            break

        # Update previous frame and corners
        prevGray = gray.copy()
        prevCorners = goodNew.reshape(-1, 1, 2)

    videoCap.release()
    cv2.destroyAllWindows()


def dimensionReductionClick(self):
    # Step 1: Convert RGB image to gray scale image
    img = cv2.imread(self.images)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Normalize gray scale image
    normalized_img = gray_img / 255.0

    # Step 3: Use PCA for dimension reduction
    w, h = gray_img.shape
    min_dim = min(w, h)
    mse_threshold = 0.1
    n = 1

    while True:
        pca = PCA(n_components=n)
        reduced_img = pca.inverse_transform(
            pca.fit_transform(normalized_img.reshape(-1, min_dim))
        )

        # Step 4 : Use MSE(Mean Square Error) to compute reconstruction error
        mse = np.mean(((normalized_img - reduced_img.reshape(w, h)) * 255.0) ** 2)

        print("n: {}, MSE: {}\n".format(n, mse))
        if mse <= mse_threshold or n >= min_dim:
            break

        n += 1

    print("Minimum n value:", n)

    # Step 5: Plot the gray scale image and the reconstruction image
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(normalized_img, cmap="gray")
    axs[0].set_title("Gray Scale Image")
    axs[0].axis("off")

    axs[1].imshow(reduced_img.reshape(w, h), cmap="gray")
    axs[1].set_title("Reconstruction Image (n={})".format(n))
    axs[1].axis("off")

    plt.show()
