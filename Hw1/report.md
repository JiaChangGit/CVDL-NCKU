# CVDL HW1 技術報告

## 1. 專案總覽

本專案是一個以 PyQt5 建立的 Computer Vision and Deep Learning 作業展示系統。它將五個主題整合在同一個 GUI 中：

1. 相機校正（Camera Calibration）
2. 擴增實境（Augmented Reality, AR）
3. 立體視覺視差圖（Stereo Disparity Map）
4. SIFT 特徵偵測與匹配（SIFT Feature Detection and Matching）
5. VGG19-BN 影像分類（Image Classification with VGG19-BN）

程式入口是 `main.py`。GUI 版面由 `qtUI.ui` 定義，`main.py` 使用 `loadUi("./qtUI.ui", self)` 載入介面，接著在 `Connect_btn()` 中把每個按鈕和對應函式連接起來。各題的核心演算法分散在 `question1.py` 到 `question5.py`，模型訓練則放在 `train.py`。

整體流程如下：

```text
使用者操作 GUI
    -> main.py 接收檔案路徑、資料夾路徑、文字輸入
    -> 呼叫 question1.py ~ question5.py 的功能函式
    -> OpenCV / PyTorch / Matplotlib 執行核心演算法
    -> 結果顯示在 OpenCV 視窗、Matplotlib 視窗或 PyQt label
```

下文依照技術背景、各題演算法、關鍵實作 trace 與 GUI 串接方式整理。

## 2. 專案架構與程式分工

### 2.1 檔案分工

| 檔案 | 責任 |
| --- | --- |
| `main.py` | GUI 主程式，負責載入 UI、保存使用者選擇的路徑、綁定按鈕事件 |
| `qtUI.ui` | Qt Designer 產生的 UI XML 檔，定義按鈕、群組、輸入框與 label |
| `Ui_qtUI.py` | UI 對應的 Python 檔，通常由 Qt 工具自動產生 |
| `question1.py` | Q1 相機校正，包含角點偵測、內參、外參、畸變、去畸變 |
| `question2.py` | Q2 AR 文字投影，將 3D 字母線段投影到棋盤影像 |
| `question3.py` | Q3 StereoBM 視差圖與點選深度估計 |
| `question4.py` | Q4 SIFT keypoints、descriptor matching、Homography 篩選 |
| `question5.py` | Q5 VGG19-BN 推論、資料增強展示、訓練曲線展示 |
| `train.py` | CIFAR-10 訓練流程，產生 `vgg19_bn.pth` 與 `5-4.png` |

### 2.2 GUI 狀態管理

`MainWindow` 物件同時保存 GUI 狀態與跨題共用資料：

```python
self.loadAllFile = ""
self.files = []
self.leftFig = ""
self.rightFig = ""
self.wide = 11
self.height = 8
self.obj_point = np.zeros((self.wide * self.height, 3), dtype=np.float32)
self.matrix00 = None
self.matrix01 = None
self.loadQ4_filename1 = ""
self.loadQ4_filename2 = ""
self.load_filenameForQ5 = ""
```

其中最重要的是：

- `self.loadAllFile`：目前選取的資料夾。例如 Q1 選 `Q1_Image/`，Q2 選 `Q2_Image/`。
- `self.files`：資料夾中的影像清單。Q1 和 Q2 都依賴這個清單逐張處理影像。
- `self.obj_point`：棋盤角點的 3D 世界座標（World Coordinates）。
- `self.matrix00`、`self.matrix01`：相機校正結果。Q1 用 `matrix00`，Q2 用 `matrix01`。

這種做法的優點是簡單：各題函式只要接收 `self` 就能讀到 GUI 狀態。缺點是 GUI 與演算法有強耦合（Tight Coupling）。例如 `question1.findCorners(self)` 不只做角點偵測，還會直接讀 `self.files`、寫 `self.matrix00`、開 OpenCV 視窗。若要寫單元測試，會比純函式困難。

## 3. 主要技術與關鍵字

### 3.1 PyQt5: 圖形化使用者介面（Graphical User Interface, GUI）

PyQt5 是 Python 對 Qt 框架的封裝。本專案使用它建立桌面應用程式。

關鍵字：

- GUI（Graphical User Interface）：圖形化使用者介面。
- Signal and Slot：Qt 的事件連接機制。按鈕被點擊時會送出 signal，對應函式是 slot。
- `QFileDialog`：檔案或資料夾選擇視窗。
- `QLabel`：顯示文字或圖片的 UI 元件。

本專案的按鈕綁定範例：

```python
self.pushButton_4.clicked.connect(self.pushButton4F)
```

Trace：

```text
使用者按下 1.1 Find Corners
    -> Qt 觸發 pushButton_4.clicked signal
    -> 呼叫 main.py 的 pushButton4F()
    -> pushButton4F() 呼叫 question1.findCorners(self)
    -> OpenCV 顯示棋盤角點結果
```

### 3.2 OpenCV: 電腦視覺函式庫（Open Source Computer Vision Library）

OpenCV 是本專案傳統電腦視覺部分的核心。使用範圍涵蓋：

- 影像讀取（Image Loading）：`cv2.imread`
- 顏色轉換（Color Conversion）：`cv2.cvtColor`
- 棋盤角點偵測（Chessboard Corner Detection）：`cv2.findChessboardCorners`
- 相機校正（Camera Calibration）：`cv2.calibrateCamera`
- 去畸變（Undistortion）：`cv2.undistort`
- 3D 到 2D 投影（3D-to-2D Projection）：`cv2.projectPoints`
- 視差估計（Disparity Estimation）：`cv2.StereoBM_create`
- SIFT 特徵（Scale-Invariant Feature Transform）：`cv2.SIFT_create`
- Homography 估計：`cv2.findHomography`

### 3.3 NumPy: 數值矩陣運算（Numerical Computing）

NumPy 用於處理影像陣列、座標點、矩陣合併。例如 Q1 建立棋盤世界座標：

```python
self.obj_point = np.zeros((self.wide * self.height, 3), dtype=np.float32)
self.obj_point[:, :2] = np.mgrid[0:self.wide, 0:self.height].T.reshape(-1, 2)
```

這段會產生 88 個 3D 點，因為棋盤內角點是 `11 x 8`。所有 z 座標都是 0，表示棋盤位於同一個平面。

若 `wide=3`、`height=2`，概念上會產生：

```text
(0,0,0), (1,0,0), (2,0,0),
(0,1,0), (1,1,0), (2,1,0)
```

本專案是 `11 x 8`，所以點更多，但概念完全相同。

### 3.4 PyTorch / Torchvision: 深度學習框架（Deep Learning Framework）

Q5 使用 PyTorch 訓練與推論 VGG19-BN。關鍵字：

- Tensor：多維陣列，是 PyTorch 計算的基本資料結構。
- Dataset：資料集抽象，例如 `torchvision.datasets.CIFAR10`。
- DataLoader：把資料切成 mini-batch 並可 shuffle。
- Model / Module：神經網路模型，繼承 `nn.Module`。
- Loss Function：損失函式，本專案使用 Cross Entropy Loss。
- Optimizer：最佳化器，本專案使用 SGD with Momentum。
- Evaluation Mode：推論模式，使用 `model.eval()`。

## 4. Q1 相機校正（Camera Calibration）

### 4.1 目標與輸入輸出

Q1 的目標是從多張棋盤影像估計相機模型。輸入是 `Q1_Image/` 中的 `.bmp` 影像，輸出包含：

- 角點偵測結果（Detected Chessboard Corners）
- 內參矩陣（Intrinsic Matrix / Camera Matrix）
- 外參矩陣（Extrinsic Matrix）
- 畸變係數（Distortion Coefficients）
- 去畸變影像（Undistorted Image）

### 4.2 相機模型（Camera Model）

針孔相機模型（Pinhole Camera Model）可以寫成：

```text
s [u v 1]^T = K [R | t] [X Y Z 1]^T
```

各符號意義：

- `(X, Y, Z)`：世界座標（World Coordinates）中的 3D 點。
- `(u, v)`：影像座標（Image Coordinates）中的 pixel 位置。
- `K`：內參矩陣（Intrinsic Matrix）。
- `R`：旋轉矩陣（Rotation Matrix）。
- `t`：平移向量（Translation Vector）。
- `[R | t]`：外參矩陣（Extrinsic Matrix）。
- `s`：尺度因子（Scale Factor），因為投影會失去絕對深度尺度。

內參矩陣形式：

```text
K =
[ fx   0  cx ]
[  0  fy  cy ]
[  0   0   1 ]
```

其中：

- `fx`、`fy`：焦距（Focal Length），單位是 pixel。
- `cx`、`cy`：主點（Principal Point），通常接近影像中心。

外參矩陣 `[R | t]` 描述棋盤相對相機的姿態（Pose）。對每張棋盤照片，外參都可能不同；但同一台相機的內參通常固定。

### 4.3 角點偵測（Chessboard Corner Detection）

在 `question1.findCorners()` 中，每張影像都會經過以下 trace：

```text
讀取影像
    -> 轉成灰階
    -> findChessboardCorners 找初始角點
    -> cornerSubPix 做 sub-pixel refinement
    -> drawChessboardCorners 畫出結果
    -> 累積 3D object points 與 2D image points
```

對應程式：

```python
img = cv2.imread(os.path.join(self.loadAllFile, file))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (self.wide, self.height), None)
if ret == True:
    new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criterias)
    img_out = cv2.drawChessboardCorners(img, (self.wide, self.height), new_corners, ret)
```

`cv2.findChessboardCorners` 會尋找棋盤內角點，不是棋盤格子的外框角落。若棋盤是 `12 x 9` 格，內角點通常會是 `11 x 8`。

`cv2.cornerSubPix` 的作用是把角點位置從整數 pixel 細化到小數 pixel。這個步驟非常重要，因為相機校正本質上是在最小化投影誤差（Reprojection Error）。角點越準，校正越穩。

舉例：若某角點真實位置是 `(320.42, 188.73)`，單純整數偵測可能只能得到 `(320, 189)`；sub-pixel refinement 會逼近小數座標，降低誤差。

### 4.4 3D Object Points 與 2D Image Points

相機校正需要多組對應點：

- Object Points：棋盤上已知的 3D 點。
- Image Points：影像上偵測到的 2D 角點。

本專案的 `self.obj_point` 是固定的棋盤座標，例如：

```text
(0,0,0), (1,0,0), ..., (10,0,0)
(0,1,0), (1,1,0), ..., (10,1,0)
...
(0,7,0), ..., (10,7,0)
```

每張影像若成功找到角點，就將同一份 `self.obj_point` 加入 `point_3D`，並將該影像偵測出的角點加入 `point_2D`：

```python
point_3D.append(self.obj_point)
point_2D.append(new_corners)
```

這代表：同一個棋盤的真實角點位置不變，但在不同照片中投影到不同 pixel 位置。

### 4.5 內參估計（Intrinsic Estimation）

核心函式：

```python
self.matrix00 = cv2.calibrateCamera(point_3D, point_2D, gray.shape[::-1], None, None)
```

OpenCV 回傳：

```text
ret, cameraMatrix, distCoeffs, rvecs, tvecs
```

本專案透過 index 取值：

- `self.matrix00[0]`：ret，通常是重投影誤差相關數值。
- `self.matrix00[1]`：camera matrix，也就是 intrinsic matrix。
- `self.matrix00[2]`：distortion coefficients。
- `self.matrix00[3]`：rotation vectors。
- `self.matrix00[4]`：translation vectors。

`findInstrinsic()` 最後印出：

```python
print("Intrinsic:", self.matrix00[1], sep="\n")
```

注意：函式名稱寫成 `findInstrinsic`，拼字比標準英文 `Intrinsic` 多了一個 `s`，但不影響程式執行。

### 4.6 外參估計（Extrinsic Estimation）

每一張棋盤影像都有自己的外參。使用者在 GUI 的 combo box 選第幾張影像後，程式會轉成 index：

```python
i = int(combox) - 1
```

OpenCV 的 rotation 以 rotation vector（旋轉向量）形式回傳。要形成 `[R | t]` 外參矩陣，需要先轉成 rotation matrix：

```python
rotation_mat = cv2.Rodrigues(self.matrix00[3][i])[0]
extrinsic_mat = np.hstack([rotation_mat, self.matrix00[4][i]])
```

關鍵字：

- Rotation Vector：旋轉向量，以 3 個數字表示旋轉軸與角度。
- Rodrigues Formula：把 rotation vector 與 rotation matrix 互相轉換的公式。
- Translation Vector：平移向量，表示世界座標原點在相機座標中的位置。

若 `R` 是 `3 x 3`，`t` 是 `3 x 1`，`np.hstack([R, t])` 會得到 `3 x 4`：

```text
[ r11 r12 r13 tx ]
[ r21 r22 r23 ty ]
[ r31 r32 r33 tz ]
```

### 4.7 畸變係數（Distortion Coefficients）

OpenCV 常見畸變模型包含 radial distortion 與 tangential distortion：

```text
k1, k2, p1, p2, k3
```

中文與英文：

- 徑向畸變（Radial Distortion）：常造成桶狀變形（Barrel Distortion）或枕狀變形（Pincushion Distortion）。
- 切向畸變（Tangential Distortion）：鏡頭與影像平面不完全平行造成的偏移。

`findDistorsion()` 會印出：

```python
print("Distorsion:", self.matrix00[2], sep="\n")
```

注意：程式中 `Distorsion` 拼字不是標準英文，標準寫法是 `Distortion`。

### 4.8 去畸變（Image Undistortion）

`showResultClick()` 先重新校正取得 camera matrix 與 distortion coefficients，再逐張影像去畸變：

```python
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(
    self.matrix00[1], self.matrix00[2], (w, h), 0, (w, h)
)
dst = cv2.undistort(img, self.matrix00[1], self.matrix00[2], None, newcameramatrix)
```

Trace：

```text
原始影像
    -> 取得原始寬高
    -> 計算 optimal new camera matrix
    -> undistort 產生去畸變影像
    -> 用 roi 裁掉無效黑邊
    -> resize 原圖與結果
    -> hstack 左右拼接
    -> imshow 顯示比較
```

`roi` 是 valid region of interest。去畸變後影像邊緣可能出現黑色無效區域，因此程式使用：

```python
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
```

把有效區域裁出來。

## 5. Q2 擴增實境（Augmented Reality, AR）

### 5.1 功能目標

Q2 的目標是在棋盤影像上畫出使用者輸入的英文文字。字母線段先定義成 3D 座標，再透過相機模型投影到影像上，因此文字會貼合棋盤角度，而不是固定在某個 pixel 位置。

流程：

```text
使用者輸入文字
    -> 讀取字母 3D 線段資料
    -> 根據字母位置加上 offset
    -> 對每張棋盤影像做 camera calibration
    -> 使用 projectPoints 做 3D-to-2D projection
    -> 使用 cv2.line 畫出投影後的字母
```

### 5.2 Calibration 的作用

如果不知道相機內參與棋盤 pose，就無法讓虛擬字母正確貼合棋盤。Q2 的 `calibaration2(self)` 會重複 Q1 的校正流程：

```python
self.matrix01 = cv2.calibrateCamera(
    point_3D, point_2D, gray.shape[::-1], None, None
)
```

`self.matrix01` 之後會提供：

- `self.matrix01[1]`：Intrinsic Matrix
- `self.matrix01[2]`：Distortion Coefficients
- `self.matrix01[3][i]`：第 i 張影像的 Rotation Vector
- `self.matrix01[4][i]`：第 i 張影像的 Translation Vector

### 5.3 字母資料庫（Alphabet Library）

字母線段資料存在：

- `Q2_Image/Q2_lib/alphabet_lib_onboard.txt`
- `Q2_Image/Q2_lib/alphabet_lib_vertical.txt`

程式用 OpenCV `FileStorage` 讀取：

```python
fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)
word.append(fs.getNode(text[i]).mat())
```

每個字母由多條 3D 線段組成。每條線段有兩個端點。例如一個簡化的字母 `I` 可能只有一條線段：

```text
[(0,0,0), (0,2,0)]
```

字母 `A` 可能由三條線段組成：

```text
left stroke:  (0,0,0) -> (1,2,0)
right stroke: (1,2,0) -> (2,0,0)
middle bar:   (0.5,1,0) -> (1.5,1,0)
```

實際資料庫中的座標由作業提供，程式只負責讀取並投影。

### 5.4 輸入文字處理

程式先將輸入轉成大寫：

```python
text = inWord.upper()
```

然後只接受英文字母：

```python
if text[i].encode("UTF-8").isalpha() and not text[i].isdigit():
    word.append(fs.getNode(text[i]).mat())
```

這代表使用者輸入 `Cvdl123` 時，實際會投影 `CVDL`。數字會被排除。

### 5.5 字母排版（Layout Offset）

程式定義六個位置：

```python
pos_adjust = [[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]]
```

這些座標都是棋盤世界座標中的 offset。對每個字母線段端點加 offset：

```python
new_axis1 = [a + b for a, b in zip(word[i][j][0], pos_adjust[i])]
new_axis2 = [a + b for a, b in zip(word[i][j][1], pos_adjust[i])]
word[i][j][0] = new_axis1
word[i][j][1] = new_axis2
```

舉例：

若某線段端點原本是：

```text
(0,0,0) -> (1,0,0)
```

第 1 個字母 offset 是 `(7,5,0)`，則變成：

```text
(7,5,0) -> (8,5,0)
```

因此字母會出現在棋盤上靠近 `(7,5)` 的位置。

### 5.6 3D 投影到 2D（3D-to-2D Projection）

核心函式：

```python
img_points, jac = cv2.projectPoints(
    axis[j],
    rotation_vector,
    transform_vector,
    self.matrix01[1],
    self.matrix01[2],
)
```

輸入：

- `axis[j]`：某個字母所有線段的 3D 點。
- `rotation_vector`：該張影像的旋轉向量。
- `transform_vector`：該張影像的平移向量。
- `self.matrix01[1]`：內參矩陣。
- `self.matrix01[2]`：畸變係數。

輸出：

- `img_points`：投影後的 2D pixel 座標。
- `jac`：Jacobian，這裡沒有使用。

最後使用 `draw()` 畫線：

```python
img = cv2.line(
    img,
    tuple(img_point[2 * i]),
    tuple(img_point[2 * i + 1]),
    (0, 0, 255),
    15
)
```

因為每條線段有兩個端點，所以第 `i` 條線段用 `2*i` 和 `2*i+1` 取出端點。

### 5.7 Q2 的重要實作 Trace

以 `Show Words on Board` 為例：

```text
main.py pushButton9F()
    -> 讀取 lineEdit 中的文字
    -> question2.horizontallyClick(self, text)
        -> calibaration2(self)
            -> 對 Q2_Image 中每張棋盤圖找角點
            -> calibrateCamera 得到 K, dist, rvecs, tvecs
        -> 讀 alphabet_lib_onboard.txt
        -> 將輸入文字轉大寫並查字母矩陣
        -> 將每個字母加上 pos_adjust
        -> 對每張影像:
            -> 取出該影像 rvec, tvec
            -> projectPoints 將字母 3D 點投影到 2D
            -> cv2.line 畫出紅色線段
            -> imshow 顯示 AR 結果
```

這裡最關鍵的是：字母座標先存在棋盤世界座標中，所以不同影像角度下，字母會自然跟著棋盤旋轉、縮放與透視變形。

## 6. Q3 立體視覺視差圖（Stereo Disparity Map）

### 6.1 功能目標

Q3 載入左右影像後，使用 StereoBM 算法建立視差圖（Disparity Map）。使用者點擊左圖時，程式會：

- 印出該點 disparity。
- 估算 depth。
- 在右圖標示估計的對應點。

### 6.2 立體視覺基本概念（Stereo Vision）

人類能感知深度，是因為左右眼看到的影像略有不同。雙相機系統也是相同概念。

關鍵字：

- Stereo Pair：左右相機拍攝的影像對。
- Rectification：校正左右影像，使對應點落在同一水平線。
- Correspondence Matching：尋找左圖某點在右圖的對應點。
- Disparity：左右對應點的 x 座標差。
- Depth：物體到相機的距離。

若左右影像已 rectified，理想情況下對應點的 y 座標相同，只需要沿 x 軸搜尋。

### 6.3 Disparity 與 Depth 的關係

基本公式：

```text
Z = fB / d
```

其中：

- `Z`：深度（Depth）
- `f`：焦距（Focal Length）
- `B`：baseline，左右相機距離
- `d`：視差（Disparity）

視差越大，物體越近；視差越小，物體越遠。

舉例：

```text
fB = 1000
d = 10  -> Z = 100
d = 50  -> Z = 20
```

同樣的相機下，`d=50` 的物體比 `d=10` 的物體更近。

本專案使用：

```python
int(342.789 * 4019.284 / (279.184 + disparity[y, x]))
```

這裡 `342.789`、`4019.284`、`279.184` 應來自作業提供的相機參數或校正後公式。分母中的 `disparity[y, x]` 越大，計算出的 depth 越小。

### 6.4 StereoBM（Block Matching）

程式使用：

```python
stereo = cv2.StereoBM_create(numDisparities=21 * 16, blockSize=19)
disparity = stereo.compute(imgL_gray, imgR_gray)
```

StereoBM 是傳統區塊比對方法。對左圖中的某一小塊區域，演算法會在右圖同一列附近找最相似的區塊。相似度通常以像素差異衡量。

參數說明：

- `numDisparities=21*16`：搜尋的最大 disparity 範圍。OpenCV 要求是 16 的倍數。
- `blockSize=19`：比對區塊大小，必須是奇數。

`blockSize` 的取捨：

- 較小 block：細節較清楚，但容易受雜訊影響。
- 較大 block：結果較平滑穩定，但物體邊界容易模糊。

### 6.5 Normalize 視差圖

StereoBM 回傳的 disparity 不一定適合直接顯示，所以程式使用：

```python
disparity = cv2.normalize(
    disparity,
    None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_8U,
)
```

這會把 disparity map 線性縮放到 0 到 255，轉成 8-bit 影像。這個步驟是為了視覺化，不是為了保留原始精準深度。換句話說，normalize 後的 disparity 適合顯示，但若要做嚴格深度估計，通常應使用原始或正確尺度的 disparity。

### 6.6 滑鼠點選 Trace

程式用 OpenCV mouse callback：

```python
cv2.setMouseCallback("imgLeft", mouse)
```

當使用者在左圖點擊：

```python
if event == cv2.EVENT_LBUTTONDOWN:
    if disparity[y, x] != 0:
        cv2.circle(
            imgR_cp,
            (x - disparity[y, x] - 50, y),
            24,
            (255, 255, 0),
            thickness=-1,
        )
```

Trace：

```text
使用者點擊左圖座標 (x, y)
    -> 讀取 disparity[y, x]
    -> 若 disparity 不為 0，代表該點有有效視差
    -> 右圖對應 x 約為 x - disparity
    -> 程式額外扣除 50 作為作業資料修正
    -> 在右圖畫圓
    -> 印出 disparity 與 depth
```

其中 `(x - disparity[y, x] - 50, y)` 是本專案的實作細節。理論上 rectified stereo 的右圖對應點是 `(x - disparity, y)`；額外 `-50` 代表程式針對此資料集加入固定偏移。

### 6.7 StereoBM 的限制

StereoBM 容易在以下情況失準：

- 沒有紋理的區域，例如白牆。
- 重複紋理，例如格子或欄杆。
- 遮擋區域，例如左圖看得到但右圖看不到的位置。
- 反光或光照差異大的區域。
- 物體邊界，因為 block 可能同時包含前景與背景。

若要改善，可使用 StereoSGBM（Semi-Global Block Matching）、調整影像 rectification、加入 speckle filtering，或使用深度學習式 stereo matching。

## 7. Q4 SIFT 特徵偵測與匹配

### 7.1 功能目標

Q4 包含兩個功能：

1. 顯示第一張影像的 SIFT keypoints。
2. 顯示兩張影像之間的 matched keypoints。

使用資料：

- `Q4_Image/Left.jpg`
- `Q4_Image/Right.jpg`

### 7.2 SIFT 是什麼

SIFT 全名是 Scale-Invariant Feature Transform，中文可稱為尺度不變特徵轉換。

關鍵字：

- Keypoint：特徵點，例如角落、斑點等局部可辨識位置。
- Descriptor：描述子，用數值向量描述 keypoint 周圍影像內容。
- Scale Invariance：尺度不變性，影像放大縮小後仍可匹配。
- Rotation Invariance：旋轉不變性，影像旋轉後仍可匹配。
- Difference of Gaussian, DoG：高斯差分，用於在不同尺度下找 extrema。

SIFT 大致流程：

```text
建立 Gaussian Scale Space
    -> 用 DoG 找候選特徵點
    -> 排除低對比與邊緣不穩定點
    -> 為 keypoint 指派主方向
    -> 產生 128 維 descriptor
```

### 7.3 Keypoint Detection Trace

`createKeyPoint(self)` 的流程：

```text
確認是否已載入影像
    -> cv2.imread 讀影像
    -> cvtColor 轉灰階
    -> SIFT_create 建立偵測器
    -> detectAndCompute 找 keypoints 和 descriptors
    -> drawKeypoints 畫出 keypoints
    -> imshow 顯示
```

程式：

```python
SIFT = cv2.SIFT_create()
key1, des1 = SIFT.detectAndCompute(img1_gray, None)
kp_image1 = cv2.drawKeypoints(img1_gray, key1, np.array([]), color=(0, 255, 0))
```

`detectAndCompute` 同時回傳：

- `key1`：keypoint list，每個 keypoint 包含位置、尺度、方向等資訊。
- `des1`：descriptor matrix。若有 N 個 keypoints，通常形狀是 `N x 128`。

### 7.4 Brute-Force Matcher

`matchedKeyPoint(self)` 使用：

```python
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
```

Brute-Force Matcher 會把第一張圖的每個 descriptor 與第二張圖的 descriptor 比較距離，找出最近的候選匹配。

`k=2` 表示對每個 descriptor 找兩個最近鄰：

- `m`：最近的 match。
- `n`：第二近的 match。

### 7.5 Lowe's Ratio Test

程式接著做 ratio test：

```python
minRatio = 0.75
for m, n in matches:
    if m.distance < minRatio * n.distance:
        goodMatches.append(m)
```

概念是：好的匹配應該明顯比第二好的匹配更接近。若 `m.distance` 和 `n.distance` 太接近，表示此 descriptor 可能不夠獨特，容易錯配。

舉例：

```text
case 1:
m.distance = 100
n.distance = 180
100 < 0.75 * 180 = 135 -> 保留

case 2:
m.distance = 100
n.distance = 120
100 < 0.75 * 120 = 90 -> 不保留
```

Case 2 中最佳匹配與次佳匹配太接近，所以不可靠。

### 7.6 Homography 與 RANSAC

若 good matches 超過 10 個，程式估計 Homography：

```python
src_pts = np.float32([key1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
dst_pts = np.float32([key2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
```

關鍵字：

- Homography：單應性矩陣，描述兩個平面之間的投影轉換。
- RANSAC（Random Sample Consensus）：隨機抽樣一致性演算法，用來排除 outliers。
- Inlier：符合模型的資料點。
- Outlier：不符合模型的錯誤資料點。

Homography 是 `3 x 3` 矩陣：

```text
[ h11 h12 h13 ]
[ h21 h22 h23 ]
[ h31 h32 h33 ]
```

它可以把第一張圖的點轉到第二張圖：

```text
s [x' y' 1]^T = H [x y 1]^T
```

RANSAC 的作用是避免錯誤匹配破壞 Homography。它會反覆隨機抽樣少量 matches 估計模型，再統計有多少 matches 符合這個模型。最後選擇 inliers 最多的模型。

### 7.7 匹配繪製

程式用 RANSAC 的 `mask` 作為 `matchesMask`：

```python
draw_params = dict(
    matchColor=(0, 255, 255),
    singlePointColor=(0, 255, 0),
    matchesMask=matchesMask,
    flags=2,
)
img3 = cv2.drawMatches(img1, key1, img2, key2, goodMatches, None, **draw_params)
```

這樣顯示的匹配線不是所有 nearest-neighbor matches，而是通過 ratio test 並符合 Homography 幾何關係的 matches。

### 7.8 Q4 重要功能 Trace

```text
main.py pushButton15F()
    -> question4.matchedKeyPoint(self)
        -> 確認兩張影像路徑存在
        -> 讀取 Left.jpg 和 Right.jpg
        -> 轉灰階
        -> SIFT.detectAndCompute 產生 keypoints/descriptors
        -> BFMatcher.knnMatch 找每個 descriptor 的兩個最近鄰
        -> 依 distance ratio 排序與篩選
        -> 若 goodMatches > 10:
            -> 取出 src_pts 和 dst_pts
            -> findHomography + RANSAC
            -> 取得 inlier mask
        -> drawMatches 畫出匹配
```

這個流程先用 descriptor 相似度找候選，再用幾何一致性篩掉錯誤匹配，是影像拼接、物件定位、平面追蹤中常見的設計。

## 8. Q5 VGG19-BN 影像分類

### 8.1 功能目標

Q5 使用 VGG19-BN 模型對 CIFAR-10 圖片分類。功能包含：

- 顯示資料增強（Data Augmentation）後的影像。
- 顯示模型結構（Model Structure）。
- 顯示訓練曲線（Training Curves）。
- 對單張影像做推論（Inference）。

CIFAR-10 類別：

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

### 8.2 CIFAR-10 資料集

CIFAR-10 是 10 類彩色影像資料集，每張影像大小是 `32 x 32 x 3`。它常用於測試 CNN 模型。

特性：

- 影像很小，所以模型需要學會從有限解析度中抓出形狀與顏色特徵。
- 類別之間有相似性，例如 cat / dog / deer / horse 都是動物。
- 背景與拍攝角度多變，因此需要資料增強提升泛化能力。

### 8.3 資料增強（Data Augmentation）

`train.py` 使用：

```python
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
```

逐項說明：

- RandomHorizontalFlip：隨機水平翻轉。對車、動物、船等類別通常合理，可增加左右方向變化。
- RandomRotation(10)：隨機旋轉最多約 10 度，使模型對小角度傾斜更穩。
- RandomCrop(32, padding=4)：先在四周補 4 pixels，再裁回 `32 x 32`，模擬物體位置平移。
- ToTensor：把 PIL image 或 NumPy array 轉成 PyTorch tensor，並將像素值縮放到 0 到 1。
- Normalize：用 CIFAR-10 常用 mean / std 標準化三個通道。

Normalize 的目的不是改變影像語意，而是讓輸入分布更適合神經網路訓練。若 RGB 值原本範圍是 0 到 1，標準化後每個通道會大致以 0 為中心，有助於梯度更新。

`question5.py` 的 `augmentClick()` 則用 Keras layers 展示資料增強視覺結果：

```python
data_augmentation1 = Sequential(
    [
        layers.RandomFlip(),
        layers.RandomRotation(10),
        layers.Rescaling(0.5),
    ]
)
```

這裡的重點是展示，不是訓練模型。實際訓練使用的是 PyTorch transforms。

### 8.4 VGG19-BN 架構

VGG19 是一種深層卷積神經網路。VGG 的設計特色是大量堆疊 `3 x 3` convolution，再用 max pooling 降低解析度。

關鍵字：

- Convolutional Layer：卷積層，提取局部特徵。
- ReLU：非線性 activation function。
- Max Pooling：降低空間尺寸，保留局部最大反應。
- Batch Normalization：批次正規化，穩定訓練。
- Classifier：分類器，通常由 fully connected layers 組成。

本專案模型：

```python
class VGG19BN(nn.Module):
    def __init__(self):
        super(VGG19BN, self).__init__()
        self.vgg19_bn = torchvision.models.vgg19_bn(num_classes=10)
        self.features = self.vgg19_bn.features
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Linear(128, 10),
        )
```

它保留 `torchvision.models.vgg19_bn` 的 feature extractor，但改寫 classifier。最後輸出 10 個 logits，對應 CIFAR-10 的 10 類。

### 8.5 Logits、Softmax 與 Cross Entropy

模型最後輸出的是 logits，不是機率。Logits 可以是任意實數，例如：

```text
[2.1, -0.3, 0.5, ..., 1.2]
```

推論時可用 softmax 轉成機率：

```python
predicted_probs = torch.softmax(outputs, dim=1)
```

Softmax 公式：

```text
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

訓練時使用 `nn.CrossEntropyLoss()`。在 PyTorch 中，CrossEntropyLoss 會內部結合 `LogSoftmax` 與 `Negative Log Likelihood Loss`，所以訓練時不需要先手動 softmax。

### 8.6 Batch Normalization

Batch Normalization 中文是批次正規化。它在訓練時對 mini-batch 的特徵做正規化：

```text
x_hat = (x - mean) / sqrt(variance + epsilon)
y = gamma * x_hat + beta
```

其中 `gamma` 和 `beta` 是可學習參數。這讓網路可以穩定訓練，同時保留表示能力。

好處：

- 降低 internal covariate shift。
- 讓梯度較穩定。
- 通常能加速收斂。
- 對初始化較不敏感。

推論時 BatchNorm 不再使用當前 batch 的 mean / variance，而是使用訓練期間累積的 running mean / running variance。因此推論前必須呼叫：

```python
net.eval()
```

### 8.7 Dropout

Dropout 是正則化（Regularization）方法。訓練時它會隨機將部分神經元輸出設為 0，避免模型過度依賴少數特徵。

本專案 classifier 中有：

```python
nn.Dropout()
```

推論時 Dropout 也必須關閉，因此同樣需要 `net.eval()`。

### 8.8 訓練流程 Trace

`train.py` 的核心 trace：

```text
設定 batch size 與 transforms
    -> 下載 / 載入 CIFAR-10 trainset 和 testset
    -> 建立 DataLoader
    -> 建立 VGG19BN 模型
    -> 設定 CrossEntropyLoss
    -> 設定 SGD optimizer
    -> for epoch in 200:
        -> model.train()
        -> 對每個 training batch:
            -> inputs, labels 移到 device
            -> optimizer.zero_grad()
            -> outputs = model(inputs)
            -> loss = criterion(outputs, labels)
            -> loss.backward()
            -> optimizer.step()
            -> 累積 train loss / train accuracy
        -> model.eval()
        -> torch.no_grad()
        -> 對 validation data 計算 val loss / val accuracy
        -> 若 valAcc > bestAcc:
            -> torch.save(model.state_dict(), "vgg19_bn.pth")
        -> 記錄 loss 和 accuracy
    -> 畫出曲線並儲存 5-4.png
```

逐行概念說明：

- `optimizer.zero_grad()`：清空上一個 batch 的梯度。PyTorch 預設梯度會累積，所以每次更新前要清掉。
- `outputs = model(inputs)`：forward pass。
- `loss.backward()`：backward pass，計算梯度。
- `optimizer.step()`：根據梯度更新權重。
- `torch.no_grad()`：驗證階段不需要梯度，可省記憶體並加速。

### 8.9 Optimizer: SGD with Momentum

本專案使用：

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
```

關鍵字：

- Learning Rate：學習率，每次更新權重的步伐大小。
- Momentum：動量，保留過去更新方向，使訓練更穩。
- Weight Decay：權重衰減，對大權重施加懲罰，降低 overfitting。

直觀例子：

如果 loss surface 像山谷，普通 SGD 可能左右震盪。Momentum 會累積主要前進方向，使更新更順。

### 8.10 Inference 推論 Trace

`question5.inferClick(self)`：

```text
建立 VGG19BN 模型
    -> torch.load 載入 vgg19_bn.pth
    -> load_state_dict 套用權重
    -> net.eval()
    -> 讀取使用者選擇的圖片
    -> transform 成 tensor
    -> unsqueeze(0) 增加 batch dimension
    -> torch.no_grad()
    -> outputs = model(image)
    -> argmax 取得類別 index
    -> GUI 顯示圖片與 predicted class
    -> softmax 取得 10 類機率
    -> Matplotlib bar chart 顯示分布
```

重點程式：

```python
checkpoint = torch.load("vgg19_bn.pth", map_location="cpu")
net.load_state_dict(checkpoint, strict=False)
net.eval()
```

`map_location="cpu"` 表示即使權重原本在 GPU 訓練，也可以載入到 CPU 推論。

`strict=False` 允許權重檔與模型有部分 key 不完全一致。這對實驗方便，但也可能掩蓋模型結構不一致問題。若追求嚴謹，應確認 missing keys 與 unexpected keys 是否合理。

`unsqueeze(0)` 的作用：

原始 transform 後 tensor shape 是：

```text
[C, H, W]
```

模型需要 batch input：

```text
[N, C, H, W]
```

所以要增加 batch dimension：

```python
image = transform(image).unsqueeze(0)
```

若圖片是 CIFAR-10 RGB，shape 會從 `[3, 32, 32]` 變成 `[1, 3, 32, 32]`。

## 9. 重要功能的整體 Trace

### 9.1 Q1 Find Intrinsic Trace

```text
GUI: Load Folder
    -> self.loadAllFile = Q1_Image
    -> self.files = sorted bmp list

GUI: 1.2 Find Intrinsic
    -> question1.findInstrinsic(self)
    -> 對每張 bmp:
        -> imread
        -> BGR to grayscale
        -> findChessboardCorners
        -> cornerSubPix
        -> append object points and image points
    -> calibrateCamera
    -> print camera matrix
```

關鍵成果是把多張影像中的 2D 角點觀測轉成相機內部參數。

### 9.2 Q2 Show Words Trace

```text
GUI: Load Folder Q2_Image
GUI: 輸入文字
GUI: Show Words on Board
    -> calibration for Q2 images
    -> read alphabet library
    -> for each character:
        -> get 3D line segments
        -> add layout offset
    -> for each image:
        -> get its rvec and tvec
        -> project 3D letter points to 2D
        -> draw red line segments
        -> show result
```

關鍵成果是使用 camera pose 讓 3D 虛擬文字貼合棋盤。

### 9.3 Q3 Disparity Trace

```text
GUI: Load Image_L
GUI: Load Image_R
GUI: Stereo Disparity Map
    -> read left and right images
    -> convert to grayscale
    -> StereoBM compute disparity
    -> normalize for display
    -> show left, right, disparity windows
    -> set mouse callback on left image
    -> click point:
        -> read disparity value
        -> estimate depth
        -> mark corresponding point on right image
```

關鍵成果是把左右影像的水平位移轉成深度提示。

### 9.4 Q4 Matched Keypoints Trace

```text
GUI: Load Image 1
GUI: Load Image 2
GUI: Matched Keypoints
    -> read images
    -> grayscale
    -> SIFT detectAndCompute
    -> BFMatcher knnMatch
    -> ratio test
    -> findHomography with RANSAC
    -> draw inlier matches
```

關鍵成果是從局部特徵相似度加上幾何一致性，得到可信匹配。

### 9.5 Q5 Inference Trace

```text
GUI: Load Image
GUI: Inference
    -> build VGG19BN
    -> load vgg19_bn.pth
    -> eval mode
    -> transform image
    -> forward pass
    -> argmax class
    -> update GUI label
    -> softmax probabilities
    -> plot class distribution
```

關鍵成果是將訓練好的 CNN 權重用於單張影像分類。

## 10. 實作限制與改進建議

### 10.1 Q1 / Q2 Calibration 重複計算

目前 Q1 和 Q2 的多個功能都會重新執行 `calibrateCamera()`。例如 Q1 的 intrinsic、extrinsic、distortion、show result 都各自重新找角點與校正。

明確問題：

- 同一批影像重複計算，浪費時間。
- 若某次偵測失敗，可能導致不同功能結果不一致。

改進：

```text
Load Folder 後先清空 cache
第一次需要 calibration 時執行並保存結果
後續功能直接讀取 cached calibration result
```

### 10.2 Q2 輸入長度

`pos_adjust` 只有六個位置，若使用者輸入超過六個英文字母，可能發生 index error。

改進方式：

- 在 GUI 限制最多 6 個字母。
- 或動態根據字串長度產生位置。
- 或超過 6 個字母時只取前 6 個並顯示提示。

### 10.3 Q3 Normalize 後的 Disparity

目前程式對 disparity 做 normalize 後，又使用 normalize 後的值估計 depth。這對視覺展示方便，但對精確深度估計不夠嚴謹。

改進：

- 保留 raw disparity 給 depth calculation。
- 另存 normalized disparity 給 visualization。

概念：

```python
raw_disparity = stereo.compute(...)
display_disparity = cv2.normalize(raw_disparity, ...)
```

### 10.4 Q4 matchesMask 風險

`matchedKeyPoint()` 中只有 `len(goodMatches) > 10` 時才會定義 `matchesMask`。若 good matches 不足，後面使用 `matchesMask` 可能造成錯誤。

改進：

```python
matchesMask = None
if len(goodMatches) > 10:
    ...
```

並在 matches 不足時顯示訊息。

### 10.5 Q5 權重檔檢查

`inferClick()` 直接讀取：

```python
torch.load("vgg19_bn.pth", map_location="cpu")
```

若目錄中沒有 `vgg19_bn.pth`，程式會錯誤中止。

改進：

```python
if not os.path.exists("vgg19_bn.pth"):
    self.label.setText("Missing model weight: vgg19_bn.pth")
    return
```

### 10.6 GUI 與演算法耦合

目前所有題目函式都接收 `self`，直接使用 GUI 物件內的狀態。這讓實作簡單，但不利於測試與重用。

改進方向：

```text
演算法函式:
    inputs: image paths, parameters
    outputs: result image, matrices, prediction

GUI 函式:
    負責讀取使用者輸入
    呼叫演算法函式
    顯示結果
```

例如 Q1 可拆成：

```python
calibrate_from_folder(folder_path, pattern_size=(11, 8)) -> CalibrationResult
```

這樣就可以不開 GUI 也測試相機校正。

## 11. 結論

本專案完整涵蓋電腦視覺與深度學習中的多個核心主題。Q1 建立相機幾何基礎，Q2 將相機幾何延伸到 AR 投影，Q3 利用左右影像差異估計深度，Q4 使用局部特徵與幾何模型完成影像匹配，Q5 則展示 CNN 在影像分類中的訓練與推論流程。

從技術關聯來看，Q1 和 Q2 是一組：相機校正提供投影模型，AR 使用投影模型把虛擬物件畫到真實影像。Q3 也是幾何視覺：它用左右視差推回深度。Q4 則展示不依賴棋盤的局部特徵匹配方法。Q5 轉向資料驅動方法，讓模型從大量影像中學習分類特徵。

因此，本專案不只是五個獨立按鈕，而是一個從幾何模型到特徵工程，再到深度學習的完整練習。
