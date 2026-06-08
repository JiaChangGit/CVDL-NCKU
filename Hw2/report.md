# CVDL Homework 2 技術報告

## 摘要

本專案整合傳統電腦視覺（Traditional Computer Vision）與深度學習（Deep Learning）兩類技術。Q1 到 Q3 使用 OpenCV 與 scikit-learn，實作背景相減（Background Subtraction）、角點偵測（Corner Detection）、光流追蹤（Optical Flow Tracking）與主成分分析（Principal Component Analysis, PCA）。Q4 使用 PyTorch 建立 VGG19-BN 風格的卷積神經網路（Convolutional Neural Network, CNN）進行 MNIST 手寫數字辨識。Q5 使用 Torchvision 的 ImageNet 預訓練 ResNet50，透過遷移學習（Transfer Learning）完成貓狗二元分類（Binary Classification），並比較隨機擦除（Random Erasing）資料增強對驗證準確率的影響。

本報告除了說明使用到的技術，也會 Trace 重要功能從 GUI 按鈕到實際演算法的執行路徑，並對關鍵參數、資料型態、模型輸入輸出與實作限制做細部探討。

## 系統架構與程式分工

本專案以 `main2.py` 作為 GUI 入口。GUI 使用 PyQt5 與 Qt Designer 設計，介面配置寫在 `cvdl2.ui`，執行時由：

```python
loadUi("./cvdl2.ui", self)
```

載入到 `MainWindow` 類別中。PyQt5 的按鈕事件使用 signal-slot 機制，也就是「使用者點擊按鈕」會觸發對應 Python function。

主要檔案分工如下：

- `main2.py`：GUI 主程式，負責載入 UI、連接按鈕、儲存使用者選取的圖片/影片路徑，並執行 Q4/Q5 推論。
- `cvdl2_Q1toQ3.py`：Q1 背景相減、Q2 光流、Q3 PCA 的傳統電腦視覺演算法。
- `cvdl2_q4_train.py`：Q4 MNIST VGG19-BN 訓練腳本。
- `ResNet50_training_withoutRE.py`：Q5 不使用 Random Erasing 的 ResNet50 訓練腳本。
- `ResNet50_training_withRE.py`：Q5 使用 Random Erasing 的 ResNet50 訓練腳本。
- `cvdl2_q5_plot.py`：產生 Q5 有無 Random Erasing 的比較圖。

整體資料流程（Data Flow）如下：

1. 使用者在 GUI 中按下 `Load Img` 或 `Load Video`。
2. 程式透過 `QFileDialog` 取得檔案路徑。
3. 圖片路徑存入 `self.images`，影片路徑存入 `self.video`。
4. 使用者按下題目功能按鈕。
5. `main2.py` 依按鈕呼叫對應 function。
6. OpenCV、PCA 或 PyTorch 模型處理資料。
7. 結果以 OpenCV 視窗、Matplotlib 圖表、Qt label 或終端機輸出呈現。

這種架構的優點是簡單直接，適合作業展示；缺點是模型類別在多個檔案中重複定義，後續維護時需要特別注意一致性。

## 重要功能 Trace：GUI 如何連到演算法

Trace 指的是沿著程式執行路徑追蹤某個功能如何被呼叫、資料如何流動、結果如何產生。以下針對本專案最重要的幾個功能做細部追蹤。

### Trace 1：載入影片並執行 Background Subtraction

使用者操作：

```text
Load Video -> 1. Background Subtraction
```

程式路徑：

```text
MainWindow.Connect_btn()
-> self.pushButton_2.clicked.connect(self.loadVideoClick)
-> self.pushButton_3.clicked.connect(self.pushButton3F)
-> pushButton3F()
-> cvdl2_Q1toQ3.subtractionClick(self)
```

關鍵實作：

```python
def loadVideoClick(self):
    self.video = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
```

此處 `QFileDialog.getOpenFileName` 會回傳一組資料，其中第 0 個元素是檔案路徑。程式將該路徑存到 `self.video`，因此後續所有影片演算法都可以使用同一個欄位取得影片。

按下背景相減按鈕後：

```python
def pushButton3F(self):
    cvdl2_Q1toQ3.subtractionClick(self)
```

這裡將 `self` 傳入 `subtractionClick`，因此 `cvdl2_Q1toQ3.py` 內的 function 可以讀到 `self.video`。這是一種簡單的跨檔案共享 GUI 狀態方式。

在 `subtractionClick` 中：

```python
videoCap = cv2.VideoCapture(self.video)
```

OpenCV 使用 `VideoCapture` 讀取影片，每次呼叫：

```python
ret, frame = videoCap.read()
```

會讀出下一個影格（frame）。`ret` 是布林值，表示是否成功讀取；`frame` 是 NumPy array，通常 shape 為：

```text
height x width x 3
```

其中 3 代表 BGR 三個色彩通道。

### Trace 2：手寫 MNIST 數字並推論

使用者操作：

```text
在黑色畫布上用滑鼠寫字 -> 3. predict
```

程式路徑：

```text
MainWindow.setGraph()
-> label_1.mousePressEvent = self.mousePress
-> label_1.mouseMoveEvent = self.mouseMove
-> predict_Q4()
-> VGG19_BN()
-> torch.load("cvdl2_Q4_vgg19_bn_model.pth")
-> model output
-> Matplotlib bar chart
```

畫布初始化：

```python
img = QImage(520, 240, QImage.Format_RGB32)
img.fill(QColor(0, 0, 0))
```

這會建立一張 520 x 240 的黑色影像。滑鼠按下或移動時，程式用 `QPainter` 在影像上畫白色線條：

```python
pen = QPen(QColor(255, 255, 255))
pen.setWidth(25)
painter.drawLine(event.pos().x(), event.pos().y(), event.pos().x(), event.pos().y())
```

注意這裡的 `drawLine` 起點與終點相同，因此實際上是畫出一個由筆刷寬度形成的點。滑鼠連續移動時，很多點連起來，看起來就像手寫筆跡。

推論時：

```python
img = self.label_1.pixmap().toImage()
img.save("q4_test.jpg")
```

Qt 畫布會被存成圖片，再由 PIL 讀回，轉換成模型需要的 tensor。這個流程雖然多了一次硬碟讀寫，但實作簡單，容易展示。

前處理（Preprocessing）：

```python
transforms.Resize((32, 32))
transforms.Grayscale(num_output_channels=1)
transforms.RandomRotation(10)
transforms.ToTensor()
```

MNIST 模型訓練時輸入是 `1 x 32 x 32`，所以 GUI 推論也必須把畫布轉成單通道 32 x 32 tensor。若尺寸或通道數不一致，第一層卷積 `Conv2d(1, 64, ...)` 會因輸入 shape 不符而失敗。

### Trace 3：載入圖片並執行 ResNet50 Cat/Dog 推論

使用者操作：

```text
Load Img -> 5.4 Inference
```

程式路徑：

```text
loadClick()
-> self.images = selected image path
-> inferenceClick()
-> Image.open(self.images).convert("RGB")
-> transform_without_erasing(image)
-> ResNet50BinaryClassifier()
-> load_state_dict(torch.load("./resnet50_noRE.pth"))
-> output.item()
-> threshold 0.5
-> label_3.setText(...)
```

推論輸入：

```python
image = Image.open(self.images).convert("RGB")
```

`convert("RGB")` 很重要，因為 ResNet50 第一層卷積預期輸入為 3-channel RGB。如果圖片是灰階、RGBA 或其他格式，不轉換就可能造成通道數錯誤。

模型輸出：

```python
output = resnet50_model(image)
```

由於最後一層是：

```text
Linear(2048, 1) -> Sigmoid
```

所以 `output.item()` 是介於 0 到 1 的數值。程式定義：

```python
if output.item() > 0.5:
    predicted_class = "Dog"
else:
    predicted_class = "Cat"
```

這裡的 0.5 是二元分類常見的 decision threshold。因為資料夾排序後 `Cat` 是類別 0，`Dog` 是類別 1，所以 sigmoid 越接近 1 越代表模型認為是 Dog。

## Q1 Background Subtraction 流程與實作

### 背景相減（Background Subtraction）

背景相減是影片前景偵測（Foreground Detection）常用方法。它的目標是從影片中分離出「正在移動或剛出現的物體」。基本想法是：

```text
目前影格 Current Frame - 背景模型 Background Model = 前景 Foreground
```

但真實場景中的背景不是固定不變的。例如：

- 光線可能改變。
- 樹葉、水面、陰影可能晃動。
- 攝影機雜訊會造成像素值微小變化。
- 車輛或行人停留太久後可能逐漸被背景模型吸收。

因此本專案使用 KNN 背景建模（K-Nearest Neighbors Background Subtraction）：

```python
subtractor = cv2.createBackgroundSubtractorKNN(
    history, distThreshold, detectShadows=True
)
```

關鍵參數說明：

- `history`：背景模型保留多少歷史影格資訊。數值越大，背景更新較慢；數值越小，背景更新較快。
- `distThreshold`：目前像素與背景樣本之間的距離門檻。門檻越小，越容易被判為前景；門檻越大，越容易被判為背景。
- `detectShadows`：是否偵測陰影。開啟後，OpenCV 可能會用特殊灰階值標示陰影區域。

本專案設定 `history = 500`，表示背景模型會參考較長時間的歷史資訊，適合交通影片這種背景相對穩定、前景物體持續移動的情境。

### 高斯模糊（Gaussian Blur）

在背景相減前，程式先做：

```python
blurredFrame = cv2.GaussianBlur(frame, (5, 5), 0)
```

Gaussian Blur 是低通濾波（Low-pass Filtering），會降低高頻細節與雜訊。使用 `(5, 5)` kernel 代表每個像素會參考周圍 5 x 5 區域。這樣做的效果是：

- 減少因影像壓縮造成的小塊雜訊。
- 讓前景遮罩比較連續。
- 降低背景模型對微小像素變化的敏感度。

代價是物體邊界會變得比較模糊。如果任務需要非常精準的輪廓，後續可加入形態學處理（Morphological Operations），例如 erosion、dilation、opening、closing。

### 遮罩（Mask）與 bitwise operation

`subtractor.apply` 產生的是單通道 mask：

```python
masks = subtractor.apply(blurredFrame)
```

mask 中每個像素代表該位置是否屬於前景。接著使用：

```python
result = cv2.bitwise_and(frame, frame, mask=masks)
```

這行的意義是：只有 mask 非 0 的位置保留原始 frame 的顏色，mask 為 0 的位置則變黑。舉例來說，如果某個像素原本是藍色車子，且 mask 判定該位置是前景，result 中就會保留該藍色；如果是背景道路，result 中就會被清為黑色。

## Q2 Optical Flow 流程與實作

### 角點偵測（Corner Detection）

Optical Flow 要追蹤的是影像中的特徵點（Feature Points）。不是所有點都適合追蹤。平坦區域沒有明顯紋理，移動一點點也看不出差異；純邊緣只有一個方向的梯度，沿著邊緣方向移動時位置容易模糊。角點則同時在兩個方向有明顯變化，因此適合追蹤。

本專案使用：

```python
corners = cv2.goodFeaturesToTrack(
    gray, maxCorners, qualityLevel, minDistance, blockSize
)
```

關鍵字中英文：

- 角點偵測（Corner Detection）
- 特徵點（Feature Point）
- Shi-Tomasi Corner Detector
- 影像梯度（Image Gradient）
- 品質門檻（Quality Level）
- 最小距離（Minimum Distance）

參數說明：

- `gray`：灰階影像。角點偵測通常不需要 RGB，使用亮度資訊即可。
- `maxCorners = 1`：最多只找 1 個角點，方便作業展示單點追蹤。
- `qualityLevel = 0.3`：角點品質至少要達到最佳角點品質的 30%。
- `minDistance = 7`：角點之間至少相距 7 pixels。
- `blockSize = 7`：計算局部梯度矩陣時使用的鄰域大小。

如果要追蹤多個物件或更穩定的軌跡，可以將 `maxCorners` 改成 50 或 100，並在每幀追蹤多個特徵點。

### 光流（Optical Flow）

光流是描述影像亮度模式在連續影格中如何移動的向量場（Vector Field）。簡單地說，如果第一幀中某個角點在 `(x, y)`，下一幀移到 `(x + dx, y + dy)`，那麼 `(dx, dy)` 就是該點的 optical flow。

Lucas-Kanade Optical Flow 建立在三個常見假設：

1. 亮度恆常（Brightness Constancy）：同一個物體點在連續影格中的亮度近似不變。
2. 小位移（Small Motion）：相鄰影格之間位移不會太大。
3. 局部一致性（Local Coherence）：鄰近像素通常屬於同一個表面，因此有相似運動。

OpenCV 使用 pyramidal Lucas-Kanade：

```python
nextCorners, status, _ = cv2.calcOpticalFlowPyrLK(
    prevGray, gray, prevCorners, None
)
```

輸入與輸出說明：

- `prevGray`：上一幀灰階影像。
- `gray`：目前幀灰階影像。
- `prevCorners`：上一幀要追蹤的點。
- `nextCorners`：估計出目前幀中對應的新位置。
- `status`：每個點是否成功追蹤。1 代表成功，0 代表失敗。

程式接著篩出成功追蹤的點：

```python
goodNew = nextCorners[status == 1]
goodOld = prevCorners[status == 1]
```

然後用新舊位置畫線：

```python
mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 100, 255), 2)
```

此處 `mask` 是累積軌跡圖。每一幀都把新的位移線段加到同一張 mask 上，因此畫面會保留歷史軌跡。最後：

```python
output = cv2.add(frame, mask)
```

把軌跡疊到目前影格。這是「即時畫面 + 歷史軌跡」的常見做法。

### 光流失敗情境

Optical Flow 可能失敗的原因包括：

- 特徵點被遮擋（Occlusion）。
- 物體移動太快，違反 small motion 假設。
- 光線劇烈改變，違反 brightness constancy。
- 特徵點離開畫面。
- 背景或物體紋理太少。

本專案只追蹤一個點，因此若該點失敗，整個追蹤就會中斷或不穩。更完整的系統通常會定期重新偵測角點，或同時追蹤多個點並過濾 outlier。

## Q3 PCA Dimension Reduction 流程與實作

### 主成分分析（Principal Component Analysis, PCA）

PCA 是一種線性降維（Linear Dimensionality Reduction）方法。它會找出資料中變異量最大的方向，稱為主成分（Principal Components）。如果資料在某些方向上的變化很小，就可以捨棄那些方向以達到壓縮效果。

以影像為例，灰階影像可視為一個矩陣。相鄰像素通常高度相關，因此影像資料有機會用較少的主成分近似重建。

本專案流程：

```python
img = cv2.imread(self.images)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
normalized_img = gray_img / 255.0
```

中英文關鍵字：

- 灰階影像（Grayscale Image）
- 正規化（Normalization）
- 降維（Dimensionality Reduction）
- 重建（Reconstruction）
- 均方誤差（Mean Squared Error, MSE）
- 主成分數（Number of Principal Components）

### PCA 壓縮與重建

程式從 `n = 1` 開始：

```python
pca = PCA(n_components=n)
reduced_img = pca.inverse_transform(
    pca.fit_transform(normalized_img.reshape(-1, min_dim))
)
```

這裡有一個重要的 shape 觀念。PCA 的輸入通常是：

```text
samples x features
```

程式將影像 reshape 成 `(-1, min_dim)`，也就是讓 PCA 將每一列視為一筆樣本，每筆樣本有 `min_dim` 個特徵。PCA 會在這個特徵空間中找出主成分。

`fit_transform` 做兩件事：

1. `fit`：學習主成分方向。
2. `transform`：把原始資料投影到低維主成分空間。

`inverse_transform` 則把低維表示轉回原始維度，因此可得到重建影像。

### MSE 停止條件

程式使用：

```python
mse = np.mean(((normalized_img - reduced_img.reshape(w, h)) * 255.0) ** 2)
```

MSE 越小，代表重建影像越接近原圖。舉例：

- MSE = 0：重建與原圖完全相同。
- MSE 很小：肉眼差異通常不明顯。
- MSE 很大：重建圖可能變模糊、邊緣消失或出現失真。

程式設定：

```python
mse_threshold = 0.1
```

因為誤差被乘回 255 尺度後再平方，`0.1` 是非常嚴格的門檻。若圖片細節很多，可能需要接近原始維度的主成分數才能達成。這也說明 PCA 對高頻紋理豐富的影像壓縮效率有限。

## Q4 MNIST Classifier Using VGG19-BN 流程與實作

### 卷積神經網路（Convolutional Neural Network, CNN）

CNN 適合處理影像，因為它利用卷積核（Convolution Kernel / Filter）在影像上滑動，學習局部特徵。淺層通常學到邊緣、角點、筆畫方向；深層會組合成更抽象的形狀，例如數字的圓弧、交叉或封閉區域。

本專案的 VGG19-BN 是 VGG-style network，但不是完整 ImageNet VGG19。原始 VGG19 很深，且輸入通常是 `3 x 224 x 224`；本專案 MNIST 是灰階數字，因此模型縮小成適合 `1 x 32 x 32` 的架構。

### 模型架構

卷積特徵萃取器（Feature Extractor）分為三個 block：

```text
Block 1:
Conv2d(1, 64, kernel_size=3, padding=1)
BatchNorm2d(64)
ReLU
Conv2d(64, 64, kernel_size=3, padding=1)
BatchNorm2d(64)
ReLU
MaxPool2d(2, 2)

Block 2:
Conv2d(64, 128, kernel_size=3, padding=1)
BatchNorm2d(128)
ReLU
Conv2d(128, 128, kernel_size=3, padding=1)
BatchNorm2d(128)
ReLU
MaxPool2d(2, 2)

Block 3:
Conv2d(128, 256, kernel_size=3, padding=1)
BatchNorm2d(256)
ReLU
Conv2d(256, 256, kernel_size=3, padding=1)
BatchNorm2d(256)
ReLU
Conv2d(256, 256, kernel_size=3, padding=1)
BatchNorm2d(256)
ReLU
MaxPool2d(2, 2)
```

重要層說明：

- 卷積層（Convolution Layer）：提取局部特徵。
- 批次正規化（Batch Normalization, BN）：穩定中間特徵分布，加速收斂。
- ReLU 激活函數（Rectified Linear Unit）：加入非線性，使模型能學習複雜映射。
- 最大池化（Max Pooling）：降低空間解析度，保留強特徵並減少計算量。
- Dropout：訓練時隨機關閉部分神經元，降低 overfitting。

### Tensor shape 範例

輸入為：

```text
batch_size x channels x height x width = 64 x 1 x 32 x 32
```

經過三次 `MaxPool2d(2, 2)` 後，空間尺寸約變成：

```text
32 -> 16 -> 8 -> 4
```

但模型接著使用：

```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
```

Adaptive Average Pooling 會把任何空間大小轉成 `1 x 1`，因此輸出成：

```text
batch_size x 256 x 1 x 1
```

再 flatten 成：

```text
batch_size x 256
```

最後進入 classifier：

```text
Linear(256, 128) -> BN -> ReLU -> Dropout -> Linear(128, 10)
```

輸出 10 個 logits，分別對應數字 0 到 9。

### 交叉熵損失（Cross Entropy Loss）

Q4 使用：

```python
criterion = nn.CrossEntropyLoss()
```

CrossEntropyLoss 適合多類別分類（Multi-class Classification）。它會將模型輸出的 logits 與正確類別 label 比較。重要的是，PyTorch 的 `CrossEntropyLoss` 內部已包含 `LogSoftmax`，因此模型最後不需要加 softmax。

舉例來說，若某張圖片是數字 7，label 為 7。模型輸出 10 個 logits：

```text
[0.1, -0.3, 0.2, 1.0, -0.5, 0.0, -0.2, 4.8, 0.7, 1.2]
```

第 7 類 logit 最大，模型就會預測為 7。訓練時 loss 會鼓勵第 7 類 logit 變大，其他類別 logit 變小。

### Q4 推論圖表的意義

GUI 推論後使用：

```python
Output = list(output[0])
for i in range(len(Output)):
    if Output[i] < 0:
        Output[i] = 0
plt.bar(labels, Output)
```

這裡畫的是修正後的 logits，不是機率（Probability）。因此長條圖可以用來比較模型偏好哪個數字，但不能直接解讀成「模型有 80% 機率認為是 8」。若要顯示機率，應使用：

```python
prob = torch.softmax(output, dim=1)
```

再將 `prob` 畫成長條圖。

## Q5 ResNet50 Cat/Dog Classification 流程與實作

### 遷移學習（Transfer Learning）

遷移學習是將在大型資料集上學到的知識轉移到新任務。ResNet50 已在 ImageNet 上學過大量物體圖片，因此前面卷積層已能偵測通用視覺特徵，例如：

- 邊緣（Edges）
- 顏色紋理（Color Textures）
- 局部形狀（Local Shapes）
- 物件部位（Object Parts）

貓狗分類不需要從零開始學所有特徵，只需要把這些通用特徵重新組合成 Cat/Dog 判斷即可。因此使用預訓練模型通常比從零訓練更快、更穩定。

### ResNet 殘差學習（Residual Learning）

深層網路可能遇到梯度消失（Vanishing Gradient）或退化問題（Degradation Problem）。ResNet 的核心是殘差連接（Residual Connection / Skip Connection）。

一般神經網路 block 直接學：

```text
H(x)
```

ResNet 改成學殘差：

```text
F(x) = H(x) - x
```

輸出為：

```text
y = F(x) + x
```

這樣的好處是，如果某一層不需要改變輸入，模型只要讓 `F(x)` 接近 0，就可以近似 identity mapping。這使深層網路比較容易最佳化。

### ResNet50BinaryClassifier 實作

本專案模型定義：

```python
resnet50_model = models.resnet50(pretrained=True)
self.features = nn.Sequential(*list(resnet50_model.children())[:-1])
self.fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(2048, 1),
    nn.Sigmoid()
)
```

`list(resnet50_model.children())[:-1]` 的意思是保留 ResNet50 除了最後 fully connected layer 以外的所有層。原本 ResNet50 的最後一層是 ImageNet 1000 類分類器，但本任務只需要 Cat/Dog 兩類，因此必須替換。

輸出維度為 1 而不是 2，是因為使用 sigmoid binary classification。兩種常見設計如下：

```text
設計 A：Linear(2048, 1) + Sigmoid + BCELoss
設計 B：Linear(2048, 2) + CrossEntropyLoss
```

本專案採用設計 A。若 output 越接近 1，代表越像 Dog；越接近 0，代表越像 Cat。

### 二元交叉熵（Binary Cross Entropy, BCE）

Q5 使用：

```python
criterion = nn.BCELoss()
```

BCE 的目標是讓正類樣本的預測值接近 1，負類樣本的預測值接近 0。對單一樣本而言：

```text
Loss = -[y log(p) + (1 - y) log(1 - p)]
```

其中：

- `y` 是真實標籤，Cat = 0，Dog = 1。
- `p` 是模型預測為 Dog 的機率。

例子：

- 真實是 Dog，`y = 1`，模型輸出 `p = 0.9`，loss 小。
- 真實是 Dog，`y = 1`，模型輸出 `p = 0.1`，loss 大。
- 真實是 Cat，`y = 0`，模型輸出 `p = 0.2`，loss 小。
- 真實是 Cat，`y = 0`，模型輸出 `p = 0.8`，loss 大。

### 資料增強（Data Augmentation）

Q5 訓練使用以下 transform：

```python
transforms.RandomResizedCrop(224)
transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)
transforms.ToTensor()
```

各方法作用：

- Random Resized Crop：隨機裁切並縮放到 224 x 224，讓模型學會不同構圖與尺度。
- Random Horizontal Flip：隨機左右翻轉，避免模型記住固定方向。
- Random Rotation：隨機旋轉 10 度，提升對姿態變化的容忍度。
- ToTensor：將 PIL image 轉成 PyTorch tensor，並把像素值轉為 `[0, 1]`。

使用 Random Erasing 的版本額外加入：

```python
transforms.RandomErasing()
```

Random Erasing 會在影像 tensor 上隨機遮掉一塊區域。它的直覺是：如果模型只靠某一小塊區域判斷，例如只看耳朵或背景，遮掉該區域後模型就會被迫學習其他特徵。這能提高模型對遮擋（Occlusion）與局部缺失的魯棒性（Robustness）。

例子：

- 原圖中狗的臉很清楚，模型可能只學會看臉。
- Random Erasing 遮住臉的一部分後，模型必須改看身體、毛色、姿態或其他線索。
- 因此在測試資料有遮擋或構圖變化時，模型可能表現更穩。

### CustomImageLoader 的目的

訓練腳本沒有直接使用 `torchvision.datasets.ImageFolder`，而是自訂：

```python
class CustomImageLoader(torch.utils.data.Dataset):
```

它的功能包括：

1. 掃描資料夾中的圖片。
2. 根據子資料夾名稱建立 class index。
3. 支援多種圖片副檔名。
4. 圖片讀取失敗時回傳黑色 placeholder image。

關鍵邏輯：

```python
for idx, target_class in enumerate(sorted(os.listdir(self.root))):
    class_to_idx[target_class] = idx
```

因為 `sorted(os.listdir(...))` 會按字母排序，目前：

```text
Cat -> 0
Dog -> 1
```

這個排序與 sigmoid threshold 的定義必須一致。如果未來資料夾名稱改變，例如改成 `0_cat`、`1_dog`，label 對應也會改變，推論解讀必須同步更新。

### 資料集不平衡（Class Imbalance）

本專案 Q5 資料量：

```text
training Cat: 5412
training Dog: 10788
validation Cat: 588
validation Dog: 1212
inference Cat: 5
inference Dog: 5
```

Dog 約為 Cat 的兩倍。這是類別不平衡（Class Imbalance）。在這種情況下，只看 accuracy 可能不夠精準。假設驗證集中 Dog 佔 67%，如果模型永遠猜 Dog，也可能得到約 67% accuracy，但 Cat 完全無法辨識。

更完整的評估指標包括：

- 混淆矩陣（Confusion Matrix）
- 精確率（Precision）
- 召回率（Recall）
- F1-score
- 每類別準確率（Per-class Accuracy）

例如，若模型常把 Cat 判成 Dog，overall accuracy 可能仍看起來不差，但 Cat recall 會很低。這對實際應用是不理想的。

### Random Erasing 結果比較

`cvdl2_q5_plot.py` 記錄：

```text
With Random Erasing: 90.7%
Without Random Erasing: 90.1%
```

Random Erasing 提升約 0.6 個百分點。這表示資料增強有正向效果，但幅度不大。可能原因：

- ResNet50 已有 ImageNet 預訓練，基礎特徵已很強。
- Crop、Flip、Rotation 已提供部分泛化能力。
- Cat/Dog 類別差異明顯，不一定高度依賴單一局部特徵。
- 只做一次訓練比較，可能受 random seed 與資料抽樣影響。

如果要更嚴謹，應進行多次訓練，回報平均值與標準差。例如：

```text
Without RE: 90.1% ± 0.4%
With RE:    90.7% ± 0.3%
```

這樣才能判斷提升是否穩定。

## 實作限制與可改進處

### 1. 推論階段不應使用隨機資料增強

目前 Q5 GUI 推論使用：

```python
transform_without_erasing = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)
```

這些 transform 在訓練時合理，但在推論時會造成同一張圖片每次結果可能不同。較標準的推論 transform 應為：

```text
Resize -> CenterCrop -> ToTensor -> Normalize
```

這能確保推論可重現（Reproducible）。

### 2. ResNet50 預訓練模型通常需要 ImageNet normalization

Torchvision ImageNet 預訓練模型通常搭配：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

目前訓練與推論都沒有做 Normalize。因為訓練與推論分布一致，模型仍可學習，但與 ImageNet 預訓練時的輸入分布不同，可能使遷移學習效果沒有完全發揮。

### 3. 權重檔格式不一致

不使用 Random Erasing 的腳本儲存：

```python
torch.save(model.state_dict(), './resnet50_noRE.pth')
```

使用 Random Erasing 的腳本儲存：

```python
torch.save({'metadata': metadata, 'model_state_dict': model.state_dict()}, './resnet50_RE.pth')
```

前者是純 state dict，後者是 checkpoint dictionary。若 GUI 要載入 `resnet50_RE.pth`，必須改成：

```python
checkpoint = torch.load("./resnet50_RE.pth")
model.load_state_dict(checkpoint["model_state_dict"])
```

否則會因格式不符而載入失敗。

### 4. 模型定義重複

`VGG19_BN` 與 `ResNet50BinaryClassifier` 在 GUI 與訓練腳本中重複定義。這會造成維護風險。例如訓練腳本改了模型架構，但 GUI 忘記同步更新，權重就可能無法載入。

較佳做法：

```text
models.py
├── VGG19_BN
└── ResNet50BinaryClassifier
```

訓練腳本與 GUI 都從 `models.py` import 同一份模型定義。

### 5. Q4 推論圖表可改為 softmax 機率

目前 Q4 bar chart 顯示的是非負 logits。更清楚的方式是顯示 softmax probability：

```python
prob = torch.softmax(output, dim=1)[0]
```

若輸出：

```text
class 3: 0.82
class 8: 0.10
class 5: 0.04
```

使用者就能清楚知道模型最相信哪一類，以及第二可能類別是誰。

## 結論

本專案完整展示了電腦視覺從傳統演算法到深度學習模型的流程。Q1 透過 KNN background subtraction 從影片中擷取移動物件；Q2 使用 Shi-Tomasi corner detection 與 pyramidal Lucas-Kanade optical flow 追蹤特徵點；Q3 使用 PCA 說明影像降維與重建誤差；Q4 使用 VGG-style CNN 完成 MNIST 多類別分類；Q5 使用 ResNet50 transfer learning 完成 Cat/Dog 二元分類，並探討 Random Erasing 對泛化能力的影響。

從實作角度看，GUI 的按鈕連接、檔案路徑儲存、OpenCV 逐幀處理、Qt 畫布轉圖片、PyTorch tensor 前處理、模型權重載入與 threshold 判斷，形成一條完整的端到端流程。若未來要提升專案品質，最重要的方向是讓推論流程 deterministic、加入 normalization、統一 checkpoint 格式、補充 confusion matrix 與 per-class metrics，並將模型架構模組化，讓訓練與 GUI 共用同一份程式碼。
