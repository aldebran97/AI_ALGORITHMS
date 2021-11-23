# 1. 准备数据集
```text
例如数据集目录是 d:/dataset，目录结构如下:

d:/dataset1 
        -cat
        -dog 
        -deer

保证每个分类有足够多的数量，最好相等
```


# 2. 将数据分为训练集和测试集

假如输入目录是d:/dataset2，训练个数占80%

```java
File baseDir=new File("d:/dataset1");
        File dstDir=new File("d:/dataset2");
        FolderNameTrainTestSplit folderNameTrainTestSplit=new FolderNameTrainTestSplit(baseDir,dstDir,w,h,0.8);
        folderNameTrainTestSplit.convert();
```

```text
结果是:

d:/dataset2 
    -train 
        -cat 
        -dog 
        -deer 
    -test 
        -cat 
        -dog 
        -deer
```


# 首次构造模型

```java
File dataset=new File("d:/dataset2");
        File mMapFile=new File("d:/tmpFile");
        long mMapFileSize=40L*1024*1024*1024;
        int w=32,h=32,channels=3,batchSize=54,seed=1;
        PictureClassification classification=new PictureClassification(
        w,h,channels,batchSize,seed,
        new File(dataset,"train"),
        new File(dataset,"test"),
        new File(dataset,"model"));
        // 激活内存映射文件文件
        classification.activateMMapFile(mMapFile,mMapFileSize);
        // 设置保存周期
        classification.setSaveInterval(10);
        // 加载数据
        classification.loadData();
        // 创建网络
        classification.buildNetwork();

```

# 加载已有模型

```java
File modelDir=new File("d:/dataset2/model");
        File mMapFile=new File("d:/tmpFile");
        long mMapFileSize=40L*1024*1024*1024;
        int loadIndex=99;
        // 从文件加载模型
        PictureClassification classification=PictureClassification.loadConfigAndModel(modelDir,loadIndex);
        // 激活内存映射文件
        classification.activateMMapFile(mMapFile,mMapFileSize);
        // 加载数据
        classification.loadData();
```

# 执行训练

```java
classification.train(100);
```

# 执行测试

```java
System.out.println(classification.evaluate().stats(true));
```

# 执行预测

```java
System.out.println(pictureClassification.predict(imgFile));
```

# 调参
只需要继承PictureClassification，重写buildNetwork()