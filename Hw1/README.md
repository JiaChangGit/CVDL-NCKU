# CVDL HW1: Computer Vision and Deep Learning GUI

本專案是 CVDL Homework 1 的整合式 PyQt5 桌面程式，將相機校正、擴增實境、立體視覺、SIFT 特徵匹配，以及 CIFAR-10 影像分類整合在同一個 GUI 中。使用者可透過介面載入作業提供的影像資料，逐題執行 OpenCV 與 PyTorch 實作結果。

## 功能總覽

| 題目 | 功能 | 主要檔案 |
| --- | --- | --- |
| Q1 | Chessboard corner detection、相機內外參與畸變係數估計、影像去畸變 | `question1.py` |
| Q2 | 將文字以 3D 線段形式投影到棋盤上，支援水平與垂直字型 | `question2.py` |
| Q3 | 使用 StereoBM 建立 disparity map，點選左圖後估計對應右圖位置與深度 | `question3.py` |
| Q4 | 使用 SIFT 偵測 keypoints，並以 BFMatcher、ratio test、RANSAC Homography 篩選匹配 | `question4.py` |
| Q5 | 使用 VGG19-BN 進行 CIFAR-10 分類，提供資料增強展示、模型結構、訓練曲線與單張推論 | `question5.py`, `train.py` |

## 專案結構

```text
Hw1/
├── main.py                  # PyQt5 GUI 入口，綁定所有按鈕事件
├── qtUI.ui                  # Qt Designer 產生的 GUI 版面
├── Ui_qtUI.py               # 由 UI 檔轉出的 Python 版面檔
├── question1.py             # Q1: camera calibration
├── question2.py             # Q2: augmented reality
├── question3.py             # Q3: stereo disparity map
├── question4.py             # Q4: SIFT
├── question5.py             # Q5: VGG19 inference and visualization
├── train.py                 # VGG19-BN on CIFAR-10 training script
├── requirements.txt         # Python 套件需求
├── 5-4.png                  # 訓練 loss / accuracy 曲線圖
├── vgg19_bn-train.png       # VGG19-BN 訓練相關圖片
├── Q1_Image/                # Q1 calibration chessboard images
├── Q2_Image/                # Q2 AR chessboard images and alphabet libraries
├── Q3_Image/                # Q3 stereo image pair
├── Q4_Image/                # Q4 SIFT matching images
└── Q5_image/                # Q5 CIFAR-10 sample images
```

## 環境需求

建議使用 Python 3.9 以上版本。此專案需要 GUI 顯示與 OpenCV 視窗，因此請在可顯示桌面視窗的環境中執行。

主要依賴：

- OpenCV / OpenCV contrib
- NumPy
- PyQt5
- Matplotlib
- PyTorch / Torchvision
- Torchsummary
- Keras / TensorFlow
- Pillow

安裝方式：

```bash
pip install -r requirements.txt
```

如果環境尚未安裝 PyTorch，請依照你的 CUDA 或 CPU 環境到 PyTorch 官方網站選擇對應安裝指令。`requirements.txt` 目前包含 `torchvision`，但沒有明確列出 `torch`。

## 執行方式

啟動 GUI：

```bash
python main.py
```

Windows PowerShell 也可使用：

```powershell
python .\main.py
```

啟動後，GUI 會提供 Load Image / Load Folder，以及 Q1 到 Q5 的功能按鈕。多數結果會透過 OpenCV 視窗、Matplotlib 視窗或 GUI label 顯示。

## 使用流程

### Q1: Camera Calibration

1. 按下 `Load Folder`，選擇 `Q1_Image/`。
2. 按下 `1.1 Find Corners`，顯示每張棋盤影像的 corner detection 結果。
3. 按下 `1.2 Find Intrinsic`，在終端機輸出 intrinsic matrix。
4. 在下拉選單選擇第幾張影像，按下 `1.3 Find Extrinsic`，輸出該張影像的 extrinsic matrix。
5. 按下 `1.4 Find Distortion`，輸出 distortion coefficients。
6. 按下 `1.5 Show Result`，顯示去畸變結果與原圖比較。

### Q2: Augmented Reality

1. 按下 `Load Folder`，選擇 `Q2_Image/`。
2. 在文字輸入框輸入英文單字。程式會轉成大寫並只保留英文字母。
3. 按下 `2.1 Show Words on Board`，將文字水平投影到棋盤平面上。
4. 按下 `2.2 Show Words on Vertically`，將文字以垂直字型投影到棋盤上。

字母線段資料來自：

- `Q2_Image/Q2_lib/alphabet_lib_onboard.txt`
- `Q2_Image/Q2_lib/alphabet_lib_vertical.txt`

### Q3: Stereo Disparity Map

1. 按下 `Load Image_L`，選擇 `Q3_Image/imL.png`。
2. 按下 `Load Image_R`，選擇 `Q3_Image/imR.png`。
3. 按下 `3.1 Stereo Disparity Map`。
4. 在左圖中點擊任一有效 disparity 的位置，終端機會輸出 disparity 與 depth，右圖會標出估計的對應點。

### Q4: SIFT

1. 按下 `Load Image 1`，選擇 `Q4_Image/Left.jpg`。
2. 按下 `Load Image 2`，選擇 `Q4_Image/Right.jpg`。
3. 按下 `4.1 Keypoints`，顯示第一張圖的 SIFT keypoints。
4. 按下 `4.2 Matched Keypoints`，顯示通過 ratio test 與 RANSAC 篩選後的匹配結果。

### Q5: VGG19

1. 按下 `Load Image`，選擇要推論的 CIFAR-10 圖片。
2. 按下 `5.1 Show Agumented Images`，選擇 `Q5_image/Q5_1/` 或 `Q5_image/Q5_4/`，顯示資料增強後的 3x3 圖片。
3. 按下 `5.2 Show Model Structure`，在終端機輸出 VGG19-BN 模型摘要。
4. 按下 `5.3 Show Acc and Loss`，顯示 `5-4.png` 訓練曲線。
5. 按下 `5.4 Inference`，對載入圖片進行分類，GUI 顯示預測類別，Matplotlib 顯示 10 類機率分布。

注意：Q5 推論需要 `vgg19_bn.pth` 權重檔。若專案目錄沒有該檔案，請先執行訓練腳本產生，或放入已訓練好的權重。

## 訓練 VGG19-BN

執行：

```bash
python train.py
```

`train.py` 會自動下載 CIFAR-10 到 `./data`，使用 VGG19-BN 架構訓練 200 epochs，並在 validation accuracy 變好時儲存最佳權重：

```text
vgg19_bn.pth
```

訓練完成後會輸出 loss / accuracy 曲線到：

```text
5-4.png
```

## 已知注意事項

- `question5.py` 的 inference 會讀取 `vgg19_bn.pth`，但此權重檔不一定包含在目前專案中。
- `train.py` 需要下載 CIFAR-10，首次執行時需要網路。
- `cv2.imshow()` 與 PyQt5 皆需要圖形化桌面環境；在純 SSH 或無顯示環境中可能無法正常開窗。
- `question4.py` 使用 `cv2.SIFT_create()`，因此需要 `opencv-contrib-python`。
- Q2 文字投影位置陣列最多配置 6 個字母位置；輸入過長時可能超出目前配置。

## 技術報告

完整技術說明、演算法原理與實作分析請參考 `report.md`。
