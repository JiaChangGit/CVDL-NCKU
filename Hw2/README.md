# CVDL Homework 2

本專案是 Computer Vision and Deep Learning Homework 2 的整合實作，內容包含傳統電腦視覺、降維分析、手寫數字分類，以及貓狗二元分類。主程式以 PyQt5 建立 GUI，並整合 OpenCV、scikit-learn PCA、PyTorch、Torchvision 與 Matplotlib，讓使用者可以透過按鈕操作影片處理、圖片分析、模型結構檢視、訓練結果展示與推論。

## 功能總覽

1. Background Subtraction
   - 載入影片後使用 OpenCV KNN background subtractor 偵測移動物件。
   - 顯示原始影格、前景遮罩與套用遮罩後的結果。

2. Optical Flow
   - Preprocessing：在影片第一幀以 `goodFeaturesToTrack` 找出角點，並用紅色十字標示。
   - Video tracking：使用 Lucas-Kanade pyramidal optical flow 追蹤角點移動軌跡。

3. PCA Dimension Reduction
   - 將輸入圖片轉成灰階並正規化。
   - 使用 PCA 做影像壓縮與重建，逐步增加主成分數直到 MSE 小於門檻。
   - 顯示原始灰階圖與重建圖。

4. MNIST Classifier Using VGG19
   - 使用自訂 VGG19-BN 類神經網路辨識 MNIST 手寫數字。
   - GUI 可顯示模型結構、訓練/驗證 loss 與 accuracy 圖。
   - 可在黑色畫布上手寫數字並進行分類，輸出各類別分數長條圖。

5. ResNet50 Cat/Dog Classification
   - 使用 ImageNet 預訓練 ResNet50 作為特徵萃取器。
   - 將最後分類層改為單一 sigmoid 輸出，進行 Cat/Dog 二元分類。
   - 比較有無 Random Erasing 資料增強的平均驗證準確率。
   - GUI 可載入圖片並顯示推論結果。

## 專案結構

```text
.
├── main2.py                         # PyQt5 GUI 主程式，串接 Q1-Q5 功能
├── Ui_cvdl2.py                      # 由 Qt Designer UI 轉出的 Python 檔
├── cvdl2.ui                         # Qt Designer UI 設計檔
├── cvdl2_Q1toQ3.py                  # Q1 background subtraction、Q2 optical flow、Q3 PCA
├── cvdl2_q4_train.py                # Q4 MNIST VGG19-BN 訓練腳本
├── ResNet50_training_withoutRE.py   # Q5 ResNet50 訓練，不使用 Random Erasing
├── ResNet50_training_withRE.py      # Q5 ResNet50 訓練，使用 Random Erasing
├── cvdl2_q5_plot.py                 # 產生 Random Erasing 比較圖
├── training_validation_plot.png     # Q4 訓練與驗證曲線
├── Comparison.png                   # Q5 Random Erasing 比較圖
├── *_train.png                      # 訓練過程或結果截圖
├── Dataset_CvDl_Hw2/
│   ├── Q1/traffic.mp4
│   ├── Q2/optical_flow.mp4
│   ├── Q3/logo.jpg
│   ├── training_dataset/Cat, Dog
│   ├── validation_dataset/Cat, Dog
│   └── inference_dataset/Cat, Dog
└── requirements.txt
```

## 環境需求

建議使用 Python 3.9 以上版本。若需要 GPU 加速，請安裝與本機 CUDA 版本相容的 PyTorch；若只執行 GUI 與 CPU 推論，也可使用 CPU 版本。

安裝套件：

```bash
pip install -r requirements.txt
```

`requirements.txt` 目前包含：

- `opencv-python`
- `numpy`
- `PyQt5`
- `matplotlib`
- `torchvision`
- `torchsummary`
- `keras`
- `tensorflow`
- `scikit-learn`
- `Pillow`

注意：訓練與推論腳本會使用 `torch`，若環境未自動安裝，請依照官方建議額外安裝 PyTorch。

## 執行 GUI

```bash
python main2.py
```

Windows PowerShell 也可使用：

```powershell
python .\main2.py
```

啟動後可使用 GUI 上的 `Load Img` 與 `Load Video` 選擇資料，再點選對應題目的按鈕執行。

## 操作說明

### Q1 Background Subtraction

1. 點選 `Load Video`。
2. 選擇 `Dataset_CvDl_Hw2/Q1/traffic.mp4` 或其他影片。
3. 點選 `1. Background Subtraction`。
4. OpenCV 會顯示三個視窗：原始 frame、mask、result。
5. 在 OpenCV 視窗中按 `q` 結束播放。

### Q2 Optical Flow

Preprocessing：

1. 點選 `Load Video`。
2. 選擇 `Dataset_CvDl_Hw2/Q2/optical_flow.mp4`。
3. 點選 `2.1 Preprocessing`。
4. 程式會在第一幀標出追蹤用角點。

Video tracking：

1. 載入同一支影片。
2. 點選 `2.2 Video tracking`。
3. 程式會以 Lucas-Kanade optical flow 追蹤角點並畫出軌跡。
4. 在 OpenCV 視窗中按 `q` 結束。

### Q3 PCA Dimension Reduction

1. 點選 `Load Img`。
2. 選擇 `Dataset_CvDl_Hw2/Q3/logo.jpg` 或其他圖片。
3. 點選 `3. Dimension Reduction`。
4. 終端機會輸出不同 `n_components` 下的 MSE。
5. Matplotlib 會顯示灰階原圖與 PCA 重建圖。

### Q4 MNIST Classifier Using VGG19

GUI 支援以下功能：

- `1. show Model Structure`：在終端機輸出 VGG19-BN 模型摘要。
- `2. Show acc and loss`：顯示 `training_validation_plot.png`。
- `3. predict`：將畫布內容存成 `q4_test.jpg`，載入 `cvdl2_Q4_vgg19_bn_model.pth` 後推論。
- `4. reset`：清空手寫畫布。

若需要重新訓練 Q4 模型：

```bash
python cvdl2_q4_train.py
```

訓練腳本會下載 MNIST 到 `./data`，訓練 60 epochs，並輸出：

- `training_validation_plot.png`
- `cvdl2_Q4_vgg19_bn_model.pth`

### Q5 ResNet50 Cat/Dog Classification

GUI 支援以下功能：

- `Load Img`：載入要推論的貓狗圖片。
- `5.1 show img`：選取包含 `Cat` 與 `Dog` 子資料夾的資料夾，顯示各一張圖片。
- `5.2 show Model Structure`：輸出 ResNet50BinaryClassifier 結構。
- `5.3 Show Comprasion`：顯示 `Comparison.png`。
- `5.4 Inference`：載入 `resnet50_noRE.pth`，對目前圖片做 Cat/Dog 推論。

重新訓練不使用 Random Erasing 的模型：

```bash
python ResNet50_training_withoutRE.py
```

重新訓練使用 Random Erasing 的模型：

```bash
python ResNet50_training_withRE.py
```

產生比較圖：

```bash
python cvdl2_q5_plot.py
```

## 資料集

專案內的 Q5 資料集結構如下：

```text
Dataset_CvDl_Hw2/
├── training_dataset/
│   ├── Cat/  # 5412 images
│   └── Dog/  # 10788 images
├── validation_dataset/
│   ├── Cat/  # 588 images
│   └── Dog/  # 1212 images
└── inference_dataset/
    ├── Cat/  # 5 images
    └── Dog/  # 5 images
```

類別標籤依照資料夾排序建立，目前 `Cat` 對應 0，`Dog` 對應 1。ResNet50 推論時輸出大於 0.5 判定為 Dog，否則判定為 Cat。

## 模型與輸出檔案

部分權重檔可能需要自行訓練產生：

- `cvdl2_Q4_vgg19_bn_model.pth`：Q4 MNIST VGG19-BN 權重。
- `resnet50_noRE.pth`：Q5 不使用 Random Erasing 的 ResNet50 權重。
- `resnet50_RE.pth`：Q5 使用 Random Erasing 的 ResNet50 權重，檔案內包含 metadata 與 `model_state_dict`。

已存在的圖片輸出：

- `training_validation_plot.png`：Q4 MNIST 訓練/驗證 loss 與 accuracy。
- `Comparison.png`：Q5 Random Erasing 比較圖，平均驗證準確率為 with Random Erasing 90.7%、without Random Erasing 90.1%。
- `cvdl2_Q4_vgg19_bn_model-train.png`、`ResNet50_training_withoutRE-train.png`、`ResNet50_training_withRE-train.png`：訓練結果截圖。

## 常見注意事項

- GUI 會透過 OpenCV 與 Matplotlib 開啟額外視窗，請在支援桌面視窗的環境執行。
- Q4 推論需要 `cvdl2_Q4_vgg19_bn_model.pth`，Q5 推論需要 `resnet50_noRE.pth`。若檔案不存在，請先執行對應訓練腳本。
- Q5 訓練預設 `num_workers=4`，在部分 Windows 或受限環境中若 DataLoader 啟動失敗，可將 `num_workers` 改小或設為 0。
- ResNet50 會載入 ImageNet 預訓練權重，第一次執行訓練或模型結構檢視時可能需要網路下載權重。
- `ResNet50_training_withRE.py` 儲存的是包含 metadata 的 checkpoint；若要在 GUI 直接載入，需要與 `main2.py` 的載入方式一致，或改用不含 Random Erasing 的 `resnet50_noRE.pth`。
