# 统计学习方法 Final Project

## 实验任务

使用 `CMU` 的人脸数据 [Neural Networks for Face Recognition (cmu.edu)](http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/faces.html) 做：

1. 使用机器学习分类算法进行识别并给出准确率
2. 使用聚类或分类算法发现表情相似的脸图

## 实验环境

`python`: `3.8` 

`sklearn`: 



## 实验过程

### 数据集解释

我们可以在这本书的 `4.7` 节看见数据集的解释。

数据集包括了 `20` 个人，每个人有 `32` 张图片，分别表示 `4` 中情绪，每种情绪都有着不同方向的图片与是否戴着墨镜，例如：

![image-20221225151404276](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221225151404276.png)

值得注意的是，图片格式为 `PGM`，这在读取图片时需要特殊的处理，我们在后面会提到。

### Task 1：分类器

在这一模块，我们会选择几个分类器模型并训练，比对其最终分类效果的差异并选择出最适合此任务的分类器模型。

#### `STEP 1` 输入与预处理

首先，我们需要知道 `pgm` 格式的文件是什么形式：

`PGM` 由两部分数据组成,分别是文件头部分和像素数据部分。

**文件头部分**

1. `.PGM`文件的格式类型(是`P2`还是`P5`)
2. 图像的宽度
3. 图像的高度
4. 图像灰度值可能的最大值

文件头的这四部分信息都是以`ASCII`码形式存储的

PGM文件头的4个信息之间用分割符分开,PGM的合法分隔符包括:空格,TAB,回车符,换行符。PGM文件头的信息应该由合法分割符号分开。

**数据部分**
数据部分记录图像每个像素的灰度值,按照图像从上到下,从左到右的顺序依次存储每个像素的灰度值。

因此，我们可以通过正则表达式来对文件头进行读取，随后根据文件头信息处理剩余的数据部分：

```python
def read_image(filename):
    with open(filename, "rb") as f:
        buffer = f.read()
    header, width, height, max_val = re.search(b"(^P5\\s(\\d+)\\s(\\d+)\\s(\\d+)\\s)", buffer).groups()
    return np.frombuffer(buffer, dtype='u1' if int(max_val) < 256 else '>u2',
                         count=int(width) * int(height), offset=len(header)
                         ).reshape((int(height), int(width)))
```

而对于所有文件，我们只需要多次调用此函数即可：

```python
def get_all_images():
    images = os.listdir("dataset/")
    all_images = []
    all_images += [i for i in images if i[-4:] == '.pgm']
    filter_images = []
    for image in all_images:
        filter_images.append([image, read_image("dataset/{0}".format(image))])
    
    return filter_images
```

随后，我们需要对数据进行预处理，由于我们需要进行的是监督学习，因此我们需要为数据打上标签，然而在这里我们可以做不同的分类任务：

1. 标签为不同的人脸
2. 标签为不同的情绪
3. 标签为不同的拍摄方向
4. 标签为是否戴墨镜

首先，我们以不同的人为分类标签，如下：

```python
def preprocessing(images, flag="name"):
    if flag == "name":
        vals = ['megak', 'night', 'glickman', 'cheyer', 'an2i', 'bpm',
                'saavik', 'kk49', 'tammo', 'steffi', 'boland', 'mitchell',
                'sz24', 'danieln', 'karyadi', 'ch4f', 'kawamura', 'phoebe',
                'at33', 'choon']
        idx = 0
    elif flag == "position":
        vals = ['left', 'right', 'up', 'straight']
        idx = 1
    elif flag == "expression":
        vals = ['neutral', 'happy', 'sad', 'angry']
        idx = 2
    elif flag == "glasses":
        vals = ['open', 'sunglasses']
        idx = 3
    else:
        raise ValueError("There is no %s flag for encoding label" % flag)

    labels = [vals.index(i[0].split("_")[idx]) for i in images]
```

然而，由于我们并不是在使用 `CNN` 等输入就是一个图像的模型，因此我们还需要对图像进行特征提取，特征提取算法包括：

1. `HOG` 特征
2. `LBP` 特征
3. `Haar` 特征

这三种算法我们均可以在 `skimage` 中直接调用：

```python
    if feature_flag == "hog":
        features = [hog(i) for i in images]
    elif feature_flag == "lbp":
        features = [local_binary_pattern(i, 7, 1.0) for i in images]
    elif feature_flag == "haar":
        features = [haar_like_feature(i, 0, 0, 5, 5) for i in images]
    else:
        raise ValueError("There is no %s flag for feature processing" % feature_flag)
```

然而，`LBP` 与 `Haar` 需要我们对参数进行调整，因此我们默认可以使用 `HOG` 进行初始的测试。

这样我们就完成了我们的预处理，并将这三个函数封装在 `utils` 包内

#### `STEP 2` 分类器的训练与测试

上文中我们默认使用 `HOG` 进行测试，由于 `HOG` 特征结合 `SVM` 分类器已经被广泛应用于图像识别中，尤其在行人检测中获得了极大的成功，因此我们首先使用 `SVM` 进行测试并将其结果作为 `baseline`。

```python
def cross_validation(train, test, model):
    scores = cross_val_score(model, train, test, cv=10)
    return scores.mean(), scores.std()


if __name__ == "__main__":
    images = utils.get_all_images()
    labels, features = utils.preprocessing(images)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.7
    )
    cls = SVC(C=1.5)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    print("Acc in test set is %.2f" % accuracy_score(y_test, pred))
    print("Cross validation acc is %.2f %.2f" % cross_validation(features, labels, cls))
```

测试结果如图所示：

![image-20221225171245807](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221225171245807.png)

随后，我们选用随机森林算法进行训练分类

```python
cls = RandomForestClassifier(n_estimators=100, criterion='entropy')
cls.fit(X_train, y_train)
```

得到结果如下：

![image-20221225172013514](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221225172013514.png)

我们尝试使用多种模型，进行对比分析，如下图所示：

![image-20221225183839074](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221225183839074.png)

如果我们将分类目标进行更改，可以得到如下图：

![image-20221225183514655](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221225183514655.png)

![image-20221225183555100](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221225183555100.png)

![image-20221225183736598](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221225183736598.png)

#### `STEP 3` 结果分析

1. 在人物分类任务中，可以发现大部分的分类器表现都较好，但决策树的表现很差，说明此分类任务特征空间分割会重叠但可分性较强，大多数分类器都能够取得较好的效果，其中以提升方法为代表的分类器较为显著。

2. 拍摄位置分类任务与是否带墨镜表现都较为一般，而其中判断是否戴墨镜是二分类任务，预计逻辑回归可能会有较好的效果，但结果并不理想，我们猜测这是因为图片为 `pgm` 格式且在提取特征后，导致眼睛部分的特征有缺失，有些拍摄位置的不同会导致眼睛部分是否带墨镜难以分辨，从而导致分类效果较差。

3. 表情的分类是效果最差的，少有分类器的准确率能够达到 `25%`，猜测是因为在通过特征提取后，由于算法精度不高，因此导致面部表情细节难以捕捉，这样会造成分类效果的极不理想，为了证实猜想，我们打印了对比图如下：

   ![image-20221227225102862](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221227225102862.png)

   ![image-20221227225116525](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221227225116525.png)

   实际上，原图我们就无法判断出表情，更不用说通过 `HOG` 特征提取后的图像了。

### Task 2：聚类与分类

#### `STEP 1` `KMeans` 聚类

首先，我们使用 `KMeans` 进行聚类并可视化，类的数量规定为 `4` ，与初始数据集的种类数一致。

![image-20221225191639379](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221225191639379.png)

将聚类结果与数据集进行对比，计算得准确率为 `24.83%`

可以发现直接进行 `KMeans` 聚类的效果并不理想，而在上面我们已经发现，分类器在表情分类任务上表现较差，所以我们在这里期望能够使用无监督学习，但又需要学习的指引。

#### `STEP 2` 半监督学习

这里我们使用半监督学习 `LabelPropagation`，并且我们将此模型与 `Task 1` 中在此任务下表现最好的 `GaussianProcessClassifier` 进行对比，我们将 `LabelPropagation` 参数设置为：

```python
model = LabelPropagation(n_neighbors=5, gamma=1.0, tol=1e-6, max_iter=10000)
```

> `LabelPropagation` 算法介绍：
>
> `LabelPropagation` 算法是一种基于标签传播的局部社区划分算法。
>
> 对于网络中的每一个节点，在初始阶段，`LabelPropagation` 算法对每一个节点一个唯一的标签，在每一个迭代的过程中，每一个节点根据与其相连的节点所属的标签改变自己的标签，更改的原则是选择与其相连的节点中所属标签最多的社区标签为自己的类别标签。随着社区标签的不断传播，最终紧密连接的节点将有共同的标签。
>
> `LabelPropagation` 算法最大的优点是其算法过程比较简单，算法速度较快。`LabelPropagation` 算法利用网络的结构指导标签的传播过程，在这个过程中无需优化任何函数。

准确率对比如下：

![image-20221225193821750](https://virgil-civil-1311056353.cos.ap-shanghai.myqcloud.com/img/image-20221225193821750.png)

#### `STEP 3` 结果分析

在 `Task 1` 中我们已经对表情分类任务进行分析，得出的结论是即使是强分类器我们也无法取得好的效果，因为原数据集在这个任务上就是难以区分的，

于是在这里，我们考察的对象是：半监督学习是否能够比无监督与有监督表现的更好。

在上图中我们可以发现，简单的半监督学习是可以通过调整参数使得效果强于监督学习的，并且半监督学习的效果好于我们在 `STEP 1` 中做的聚类，并且速度上也并不比无监督慢。

个人拙见是半监督学习是一种准确率与效率统筹兼顾的学习方法，但我们无法一次就获得较高的准确率，这要求我们花更多的时间在超参数的调整上，但相比于较少的训练时间，我认为这是值得的。

## 总结



通过两个 `Task`，让我对机器学习算法的应用有着更深入的了解，较为遗憾的是本学期只手动实现了少数几个算法，并未实现太过复杂的算法，因此在项目中只能调包进行应用，以让自己有一些粗浅的认知。

通过使用算法，分析对比结果，让我对数据特征处理，算法的选择，超参数的调整等任务有着进一步的认识。
