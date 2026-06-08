# Hw1 圖解導覽

這份是給剛接手 Hw1 的人看的。先不用急著看每一行程式，照著下面這些圖走，會比較快知道按鈕按下去之後，資料從哪裡來、跑過哪些 OpenCV / PyTorch API、最後又顯示到哪裡。

## 先看整體

Hw1 是一個 PyQt5 GUI。`main.py` 負責視窗、載檔、接按鈕；`question1.py` 到 `question5.py` 才是真正做事的地方。`train.py` 是離線訓練腳本，不是 GUI 啟動時會自動跑的東西。

```mermaid
flowchart LR
    U[使用者] --> GUI[main.py / MainWindow]
    GUI --> UI[qtUI.ui]
    GUI --> Q1[question1.py<br/>相機校正]
    GUI --> Q2[question2.py<br/>AR 字母投影]
    GUI --> Q3[question3.py<br/>雙目 disparity]
    GUI --> Q4[question4.py<br/>SIFT 特徵配對]
    GUI --> Q5[question5.py<br/>CIFAR-10 / VGG19 推論]
    Train[train.py<br/>離線訓練] --> PTH[vgg19_bn.pth]
    Train --> Plot[5-4.png]
    PTH --> Q5
    Plot --> Q5
```

```mermaid
flowchart TB
    Root[Hw1] --> Main[main.py]
    Root --> UIFile[qtUI.ui / Ui_qtUI.py]
    Root --> Q1Img[Q1_Image<br/>棋盤格校正圖]
    Root --> Q2Img[Q2_Image<br/>棋盤格 + 字母座標庫]
    Root --> Q3Img[Q3_Image<br/>imL / imR]
    Root --> Q4Img[Q4_Image<br/>Left / Right]
    Root --> Q5Img[Q5_image<br/>CIFAR 範例圖]
    Root --> Model[vgg19_bn.pth<br/>訓練後權重]
    Root --> Result[5-4.png<br/>訓練曲線]
```

```mermaid
sequenceDiagram
    actor User as 使用者
    participant App as QApplication
    participant W as MainWindow
    participant UI as qtUI.ui
    participant Qt as Qt 事件迴圈

    User->>App: python main.py
    App->>W: 建立 MainWindow
    W->>UI: loadUi("./qtUI.ui", self)
    W->>W: 初始化路徑、棋盤格 3D 點、矩陣欄位
    W->>W: Connect_btn()
    W->>Qt: window.show() / app.exec_()
    Qt-->>User: 等使用者按按鈕
```

## GUI 訊號怎麼接

這張圖很重要。Hw1 不是每個按鈕都自己算，它大多只是把事件轉去對應的 question 函式。

```mermaid
flowchart TB
    subgraph Load[載入資料]
        B1[pushButton_1<br/>選資料夾] --> F1[pushButton1F<br/>整理 bmp 檔名]
        B2[pushButton_2<br/>選左圖] --> F2[pushButton2F<br/>leftFig]
        B3[pushButton_3<br/>選右圖] --> F3[pushButton3F<br/>rightFig]
        B12[pushButton_12<br/>Q4 圖 1] --> F12[pushButton12F]
        B13[pushButton_13<br/>Q4 圖 2] --> F13[pushButton13F]
        B16[pushButton_16<br/>Q5 圖] --> F16[question5.loadClick]
    end

    subgraph Work[功能按鈕]
        B4[Find Corners] --> Q1a[question1.findCorners]
        B5[Find Intrinsic] --> Q1b[question1.findInstrinsic]
        BE[Find Extrinsic] --> Q1c[question1.findExtrinsic]
        B7[Find Distortion] --> Q1d[question1.findDistorsion]
        B8[Show Result] --> Q1e[question1.showResultClick]
        B9[Show Words on Board] --> Q2a[question2.horizontallyClick]
        B10[Show Words Vertically] --> Q2b[question2.verticallyClick]
        B11[Stereo Disparity] --> Q3[question3.disparityMap]
        B14[Keypoints] --> Q4a[question4.createKeyPoint]
        B15[Matched Keypoints] --> Q4b[question4.matchedKeyPoint]
        B17[Augment] --> Q5a[question5.augmentClick]
        B18[Structure] --> Q5b[question5.structClick]
        B19[Acc] --> Q5c[question5.accClick]
        B20[Inference] --> Q5d[question5.inferClick]
    end
```

```mermaid
stateDiagram-v2
    [*] --> 空白視窗
    空白視窗 --> 已選資料夾: pushButton1F
    空白視窗 --> 已選左右圖: pushButton2F / pushButton3F
    空白視窗 --> 已選Q4圖: pushButton12F / pushButton13F
    空白視窗 --> 已選Q5圖: question5.loadClick
    已選資料夾 --> Q1可跑: Q1_Image
    已選資料夾 --> Q2可跑: Q2_Image
    已選左右圖 --> Q3可跑
    已選Q4圖 --> Q4可跑
    已選Q5圖 --> Q5可推論
    Q1可跑 --> 顯示OpenCV視窗
    Q2可跑 --> 顯示AR視窗
    Q3可跑 --> 顯示disparity與左右圖
    Q4可跑 --> 顯示Keypoints或Matched
    Q5可推論 --> GUI label顯示類別與圖片
```

## Q1 相機校正

Q1 的核心是「棋盤格角點」。`main.py` 在初始化時先建立 11 x 8 的世界座標點，之後每張圖都找 2D 角點，再拿 3D / 2D 配對去做 `cv2.calibrateCamera()`。

```mermaid
flowchart LR
    Folder[Q1_Image 資料夾] --> BMP[排序後的 bmp 清單]
    BMP --> Read[cv2.imread]
    Read --> Gray[cv2.cvtColor BGR to GRAY]
    Gray --> Find[cv2.findChessboardCorners]
    Find --> Refine[cv2.cornerSubPix]
    Refine --> Pair[收集 obj_point 與 image points]
    Pair --> Calib[cv2.calibrateCamera]
    Calib --> K[Intrinsic Matrix]
    Calib --> D[Distortion Coefficients]
    Calib --> R[r_vecs]
    Calib --> T[t_vecs]
```

```mermaid
sequenceDiagram
    actor User as 使用者
    participant GUI as main.py
    participant Q1 as question1.py
    participant CV as OpenCV

    User->>GUI: 選 Q1_Image 資料夾
    GUI->>GUI: os.listdir + 依數字排序 bmp
    User->>GUI: 按 Find Corners
    GUI->>Q1: findCorners(self)
    loop 每張棋盤格圖
        Q1->>CV: imread / cvtColor
        Q1->>CV: findChessboardCorners
        Q1->>CV: cornerSubPix
        Q1->>CV: drawChessboardCorners / imshow
    end
    Q1->>CV: calibrateCamera(point_3D, point_2D)
    CV-->>Q1: ret, intrinsic, distortion, r_vecs, t_vecs
    Q1-->>GUI: 寫回 self.matrix00
```

```mermaid
flowchart TB
    Obj[self.obj_point<br/>88 個 3D 點<br/>z = 0] --> Pair
    Img[每張圖的 new_corners<br/>88 個 2D 點] --> Pair[3D / 2D 對應]
    Pair --> Calib[calibrateCamera]
    Calib --> M0[self.matrix00]
    M0 --> Ret[0 ret]
    M0 --> Intrinsic[1 intrinsic]
    M0 --> Distortion[2 distortion]
    M0 --> Rvecs[3 r_vecs]
    M0 --> Tvecs[4 t_vecs]
```

```mermaid
flowchart LR
    Combo[find_extrinsic_combobox<br/>第幾張圖] --> Index[i = int(text) - 1]
    Index --> Rvec["self.matrix00[3][i]"]
    Rvec --> Rodrigues[cv2.Rodrigues]
    Rodrigues --> R[3 x 3 rotation matrix]
    Index --> Tvec["self.matrix00[4][i]"]
    R --> Stack[np.hstack]
    Tvec --> Stack
    Stack --> Extrinsic[3 x 4 extrinsic matrix]
```

```mermaid
flowchart LR
    Img[原圖] --> HWD[讀取 h, w]
    Calib[calibrateCamera 結果] --> NewK[getOptimalNewCameraMatrix]
    HWD --> NewK
    Img --> Undistort[cv2.undistort]
    NewK --> Undistort
    Undistort --> ROI[依 roi 裁切]
    ROI --> Resize1[resize 480 x 480]
    Img --> Resize2[resize 480 x 480]
    Resize1 --> Stack[np.hstack]
    Resize2 --> Stack
    Stack --> Show[imshow undistorted result]
```

## Q2 AR 字母投影

Q2 其實是 Q1 校正結果的延伸。先重新校正得到相機矩陣，讀文字的 3D 線段座標，再用 `cv2.projectPoints()` 把 3D 線段投到每張棋盤格上。

```mermaid
flowchart TB
    Input[LineEdit 文字] --> Upper[轉大寫]
    Upper --> Filter[只收英文字母]
    Filter --> Lib{水平或垂直?}
    Lib -->|水平| OnBoard[alphabet_lib_onboard.txt]
    Lib -->|垂直| Vertical[alphabet_lib_vertical.txt]
    OnBoard --> Nodes[cv2.FileStorage getNode]
    Vertical --> Nodes
    Nodes --> Segments[每個字母的 3D 線段]
    Segments --> Offset[pos_adjust<br/>最多 6 個字母位置]
    Offset --> Project[cv2.projectPoints]
    Project --> Draw[cv2.line 畫到棋盤格圖]
```

```mermaid
sequenceDiagram
    actor User as 使用者
    participant GUI as main.py
    participant Q2 as question2.py
    participant CV as OpenCV
    participant FS as 字母座標檔

    User->>GUI: 輸入文字
    User->>GUI: 按水平或垂直顯示
    GUI->>Q2: horizontallyClick / verticallyClick
    Q2->>Q2: calibaration2(self)
    Q2->>CV: calibrateCamera
    Q2->>FS: 讀 alphabet_lib_onboard 或 vertical
    loop 每個合法字母
        Q2->>FS: getNode(letter).mat()
        Q2->>Q2: 加上 pos_adjust
    end
    loop 每張棋盤格圖
        Q2->>CV: Rodrigues / findChessboardCorners
        Q2->>CV: projectPoints
        Q2->>CV: line / imshow
    end
```

```mermaid
flowchart LR
    Letter[A] --> Lines[字母庫裡的 3D 線段]
    Lines --> Offset1[加第 1 格偏移]
    Offset1 --> Axis[axis: N x 3]
    Axis --> Cam[用 r_vec, t_vec, intrinsic, distortion]
    Cam --> ImgPts[投影成 2D img_points]
    ImgPts --> DrawLine[每兩點畫一條紅線]
```

## Q3 雙目 Disparity 和深度

Q3 要先選左圖與右圖。程式用 StereoBM 算 disparity，使用者在左圖點一下時，拿該點的 disparity 去估深度，並在右圖畫出對應位置。

```mermaid
flowchart LR
    Left[左圖 imL] --> GrayL[BGR to GRAY]
    Right[右圖 imR] --> GrayR[BGR to GRAY]
    GrayL --> Stereo[cv2.StereoBM_create<br/>numDisparities=336<br/>blockSize=19]
    GrayR --> Stereo
    Stereo --> Raw[stereo.compute]
    Raw --> Norm[cv2.normalize 0-255]
    Norm --> Display[disparity 視窗]
```

```mermaid
sequenceDiagram
    actor User as 使用者
    participant GUI as main.py
    participant Q3 as question3.py
    participant CV as OpenCV

    User->>GUI: Load Image_L
    GUI->>GUI: self.leftFig = path
    User->>GUI: Load Image_R
    GUI->>GUI: self.rightFig = path
    User->>GUI: 按 Stereo Disparity Map
    GUI->>Q3: disparityMap(self)
    Q3->>CV: imread 左右圖
    Q3->>CV: StereoBM.compute
    Q3->>CV: imshow disparity / imgLeft / imgRight
    User->>CV: 在 imgLeft 點一下
    CV->>Q3: mouse callback(x, y)
    Q3->>Q3: depth = 342.789 * 4019.284 / (279.184 + disparity)
    Q3->>CV: 在右圖畫對應圓點
```

```mermaid
flowchart TB
    Click[(左圖點擊 x,y)] --> D{"disparity[y,x] != 0?"}
    D -->|否| Ignore[不處理]
    D -->|是| Xr[x - disparity - 50]
    Xr --> Circle[右圖畫圓]
    D --> Depth[計算 depth]
    Depth --> Print[console 印 disparity / depth]
```

## Q4 SIFT 特徵與配對

Q4 分兩段：先只看單張圖的 keypoints；再看兩張圖的配對。真正篩掉爛配對的是 ratio test，然後用 RANSAC 找 homography 的 mask。

```mermaid
flowchart LR
    Img1[圖 1] --> Gray1[灰階]
    Gray1 --> SIFT1[SIFT.detectAndCompute]
    SIFT1 --> Keypoints[Keypoints]
    Keypoints --> Draw[cv2.drawKeypoints]
    Draw --> Show[imshow Keypoints]
```

```mermaid
flowchart TB
    L[Left.jpg] --> GL[灰階]
    R[Right.jpg] --> GR[灰階]
    GL --> S1[SIFT key1, des1]
    GR --> S2[SIFT key2, des2]
    S1 --> Match[BFMatcher.knnMatch k=2]
    S2 --> Match
    Match --> Sort[依 m.distance / n.distance 排序]
    Sort --> Ratio[ratio test<br/>m.distance < 0.75 * n.distance]
    Ratio --> Good[goodMatches]
    Good --> Enough{數量 > 10?}
    Enough -->|是| H[findHomography RANSAC]
    H --> Mask[matchesMask]
    Mask --> DrawMatch[cv2.drawMatches]
    Enough -->|否| DrawMatch
    DrawMatch --> Show[imshow Matched]
```

```mermaid
sequenceDiagram
    actor User as 使用者
    participant GUI as main.py
    participant Q4 as question4.py
    participant CV as OpenCV

    User->>GUI: Load Image 1
    User->>GUI: Load Image 2
    User->>GUI: 按 Matched Keypoints
    GUI->>Q4: matchedKeyPoint(self)
    Q4->>CV: SIFT_create
    Q4->>CV: detectAndCompute for img1
    Q4->>CV: detectAndCompute for img2
    Q4->>CV: BFMatcher.knnMatch(k=2)
    Q4->>Q4: ratio test
    Q4->>CV: findHomography(..., RANSAC, 10)
    Q4->>CV: drawMatches(..., matchesMask)
```

## Q5 CIFAR-10 / VGG19

Q5 有兩條線：`train.py` 先把 CIFAR-10 訓練成 `vgg19_bn.pth`，GUI 的 `question5.inferClick()` 再載入這個權重做推論。也就是說，按 GUI 不會訓練模型，只會使用已經訓練好的檔案。

```mermaid
flowchart TB
    subgraph Train[離線訓練 train.py]
        CIFAR[CIFAR-10 train/test] --> Aug[RandomFlip / Rotation / Crop / Normalize]
        Aug --> Loader[DataLoader batch=16]
        Loader --> VGG[VGG19BN]
        VGG --> Loss[CrossEntropyLoss]
        Loss --> SGD[SGD lr=0.01 momentum=0.9]
        SGD --> Loop[200 epochs]
        Loop --> Best[best validation acc]
        Best --> PTH[vgg19_bn.pth]
        Loop --> Curve[5-4.png]
    end

    subgraph GUI[GUI 推論 question5.py]
        Input[使用者選圖] --> Load[load_filenameForQ5]
        Load --> Infer[inferClick]
        PTH --> Infer
        Infer --> Label[GUI label 顯示類別]
        Infer --> Bar[Matplotlib 類別分布]
    end
```

```mermaid
flowchart LR
    Img[Q5 單張圖片] --> T1[RandomHorizontalFlip]
    T1 --> T2[RandomRotation 10]
    T2 --> T3[RandomCrop 32 padding 4]
    T3 --> T4[ToTensor]
    T4 --> T5[Normalize CIFAR mean/std]
    T5 --> Batch[unsqueeze batch 維度]
    Batch --> Model[VGG19BN]
    Model --> Softmax[softmax 10 類]
    Softmax --> Argmax[最大值類別]
```

```mermaid
flowchart TB
    VGG[VGG19BN] --> Base[torchvision.models.vgg19_bn<br/>features]
    Base --> Flat[view batch x 512]
    Flat --> FC1[Linear 512 to 128]
    FC1 --> ReLU[ReLU]
    ReLU --> BN[BatchNorm1d 128]
    BN --> Drop[Dropout]
    Drop --> FC2[Linear 128 to 10]
    FC2 --> Logits[10 類 logits]
```

```mermaid
sequenceDiagram
    actor User as 使用者
    participant GUI as main.py
    participant Q5 as question5.py
    participant Torch as PyTorch
    participant MPL as Matplotlib

    User->>GUI: Load Image
    GUI->>Q5: loadClick(self)
    Q5-->>GUI: self.load_filenameForQ5
    User->>GUI: 按 Inference
    GUI->>Q5: inferClick(self)
    Q5->>Torch: 建立 VGG19BN
    Q5->>Torch: torch.load("vgg19_bn.pth")
    Q5->>Torch: model.eval()
    Q5->>Torch: forward(image)
    Torch-->>Q5: logits
    Q5->>GUI: label 顯示 Predicted
    Q5->>MPL: bar 顯示 10 類機率
```

```mermaid
flowchart TB
    Folder[Q5_image/Q5_1 或 Q5_4] --> PNG[讀 png 檔清單]
    PNG --> Grid[3 x 3 subplot]
    Grid --> Each[逐張 Image.open]
    Each --> KerasAug[Keras RandomFlip / RandomRotation / Rescaling]
    KerasAug --> Show[plt.show 增強後圖片]
```

## API 呼叫圖

這張可以拿來快速查「某題到底靠哪個外部 API」。

```mermaid
flowchart TB
    Hw1[Hw1] --> PyQt[PyQt5<br/>QApplication / QMainWindow / QFileDialog / loadUi]
    Hw1 --> OpenCV[OpenCV]
    Hw1 --> Torch[PyTorch / Torchvision]
    Hw1 --> MPL[Matplotlib]
    Hw1 --> Keras[Keras augmentation]

    OpenCV --> C1[findChessboardCorners / cornerSubPix / calibrateCamera]
    OpenCV --> C2[projectPoints / Rodrigues]
    OpenCV --> C3[StereoBM_create / normalize / setMouseCallback]
    OpenCV --> C4[SIFT_create / BFMatcher / findHomography]
    OpenCV --> C5[imshow / waitKey / destroyAllWindows]

    Torch --> T1[torchvision.datasets.CIFAR10]
    Torch --> T2[torchvision.models.vgg19_bn]
    Torch --> T3[DataLoader / CrossEntropyLoss / SGD]
    Torch --> T4[torch.save / torch.load]
```

## 題目之間的依賴關係

Q1 和 Q2 都吃棋盤格，而且 Q2 其實也要做一次校正；Q3/Q4/Q5 幾乎是各自獨立。

```mermaid
flowchart LR
    Chess[棋盤格影像] --> Q1[Q1 校正]
    Chess --> Q2[Q2 AR 投影]
    Alpha[字母座標庫] --> Q2
    Stereo[左右影像] --> Q3[Q3 disparity]
    Pair[兩張一般影像] --> Q4[Q4 SIFT]
    CIFAR[CIFAR-10 / 範例圖] --> Q5[Q5 VGG19]
    Train[train.py] --> Q5
```

## 常見問題點

```mermaid
flowchart TB
    A[按功能前沒先載檔] --> B[路徑欄位是空字串]
    B --> C[函式直接 return 或 cv2.imread 失敗]
    D[缺 vgg19_bn.pth] --> E[Q5 inference torch.load 失敗]
    F[OpenCV 視窗沒關] --> G[waitKey 阻塞 GUI 體感]
    H[Q2 字太長] --> I[pos_adjust 只準備 6 個位置]
    J[沒裝 opencv-contrib] --> K[SIFT_create 可能不能用]
```

## Hw1 模組閱讀順序

這段整理各模組的依賴與資料流，方便快速掌握專案架構。

```mermaid
flowchart LR
    M1[GUI 架構] --> M2[Q1 校正資料流]
    M2 --> M3[Q2/Q3 幾何結果視覺化]
    M3 --> M4[Q4 特徵配對與篩選]
    M4 --> M5[Q5 訓練產物與 GUI 推論]
```

整理如下：

1. `main.py` 是 GUI 入口，真正的工作分給 `question1` 到 `question5`；`train.py` 是另一條離線訓練線。
2. Q1 的資料流是棋盤格影像進來，找 2D 角點，搭配固定的 3D 棋盤座標，最後得到 intrinsic、distortion、r_vecs、t_vecs。
3. Q2/Q3 分別把校正結果拿來做投影，並從左右圖的 disparity 估深度，本質上都是座標轉換。
4. Q4 先取 SIFT 特徵，再用 BFMatcher 找候選，ratio test 篩一次，RANSAC 再篩一次。
5. Q5 的離線訓練會產生 `vgg19_bn.pth` 和 `5-4.png`，GUI 只負責載權重、做 transform、跑 forward、顯示結果。

整體來看，Hw1 是一個用 PyQt5 把五個小型 Computer Vision / Deep Learning demo 串起來的專案；主程式管互動，各題模組管演算法，深度學習的權重則由訓練腳本先準備好。
