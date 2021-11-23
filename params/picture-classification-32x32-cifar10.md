### 情况
训练集大小 39497 测试集大小 11358 <br>
分类数10 像素32x32 <br>
芯片apple m1 <br>
系统 macos big sur 11.6

### 7周期结果

```text
epoch: 7


========================Evaluation Metrics========================
 # of classes:    10
 Accuracy:        0.7544
 Precision:       0.7616
 Recall:          0.7541
 F1 Score:        0.7558
Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)


=========================Confusion Matrix=========================
    0    1    2    3    4    5    6    7    8    9
---------------------------------------------------
  900    3   74   41   22    9    5    7   37   31 | 0 = airplane
   23  878    7   12    5    0   11    4   39  160 | 1 = automobile
   56    1  792   79   84   29   51   29   11    7 | 2 = bird
   18    5   61  712   47  123   57   52   18   15 | 3 = cat
   15    0   84   78  807   35   42   69   11    6 | 4 = deer
    7    0   52  231   39  722   19   55    5    3 | 5 = dog
    9    3   47   78   50   22  941    6    6    8 | 6 = frog
   14    1   41   53   89   47   10  868    2   15 | 7 = horse
   89    9    8   16   10    4    8    4  930   24 | 8 = ship
   39   20    7   21    2    5    7    6   26 1018 | 9 = truck

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================```
```

### 网络结构

```java
    public void buildNetwork() {

        Activation activation = Activation.RELU;
        LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

        ConvolutionLayer cnn1 = new ConvolutionLayer.Builder(3, 3).nIn(channels).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(64).activation(activation).build();

        ConvolutionLayer cnn1_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(64).activation(activation).build();

        SubsamplingLayer pool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();


        ConvolutionLayer cnn2 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(128).activation(activation).build();

        ConvolutionLayer cnn2_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(128).activation(activation).build();


        SubsamplingLayer pool2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();

        ConvolutionLayer cnn3 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(256).activation(activation).build();

        ConvolutionLayer cnn3_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(256).activation(activation).build();


        SubsamplingLayer pool3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();


        LSTM lstm = new LSTM.Builder().activation(activation)
                .nOut(512).build();

        LSTM lstm2 = new LSTM.Builder().activation(activation)
                .nOut(512).build();


//        DenseLayer denseLayer = new DenseLayer.Builder().activation(activation)
//                .nOut(512).build();
//
//        DenseLayer denseLayer2 = new DenseLayer.Builder().activation(activation)
//                .nOut(256).build();
//
//        DenseLayer denseLayer3 = new DenseLayer.Builder().activation(activation)
//                .nOut(128).build();


        OutputLayer outputLayer = new OutputLayer.Builder(lossFunction)
                .nOut(nLabels).activation(Activation.SOFTMAX).build();

        int i = 0;

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().seed(seed)
                .l1(0)
                .l2(0)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(i++, cnn1)
                .layer(i++, cnn1_1)
                .layer(i++, pool1)
                .layer(i++, cnn2)
                .layer(i++, cnn2_1)
                .layer(i++, pool2)
                .layer(i++, cnn3)
                .layer(i++, cnn3_1)
                .layer(i++, pool3)
                .layer(i++, lstm)
                .layer(i++, lstm2)
                .layer(i++, outputLayer)
                .setInputType(InputType.convolutional(h, w, channels))
                .build();

        network = new MultiLayerNetwork(configuration);
        network.init();
    }
```
### 初始化

```java
 PictureClassification classification = new PictureClassification(
                32, 32, 3, 100, 1,
                new File(trainDir),
                new File(testDir),
                new File(modelDir));
```