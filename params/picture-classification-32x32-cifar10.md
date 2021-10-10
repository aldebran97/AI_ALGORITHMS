### 1. 训练集大小 39552 测试集大小 11365 分类数10 像素32*32


### 2. 第6周期结果:

```text
========================Evaluation Metrics========================
 # of classes:    10
 Accuracy:        0.7093
 Precision:       0.7106
 Recall:          0.7101
 F1 Score:        0.7054
Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)


=========================Confusion Matrix=========================
    0    1    2    3    4    5    6    7    8    9
---------------------------------------------------
  937   15   48    9   16   15   17   16   54   36 | 0 = airplane
   27  932    4    4    8    5   16    7   17  105 | 1 = automobile
   82    6  645   36   90   62  125   41   12    5 | 2 = bird
   35   12   96  471   88  244  178   51   12   15 | 3 = cat
   30    5   78   29  746   27  134   51   10    9 | 4 = deer
    8    8   87  130   74  655   74   63    6    6 | 5 = dog
    8    8   29   42   18   20 1020    6    5    6 | 6 = frog
   15    3   41   27  103   61   27  846    2    5 | 7 = horse
   96   40   23    6    8    7   11    3  874   40 | 8 = ship
   51   67   10    7   15   12   10   10   24  935 | 9 = truck

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================

```

### 3. 网络结构

```java
    public void buildNetwork() {

        Activation activation = Activation.TANH;
        LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

        ConvolutionLayer cnn1 = new ConvolutionLayer.Builder(3, 3).nIn(channels).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(32).activation(activation).build();

        ConvolutionLayer cnn1_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(32).activation(activation).build();

        SubsamplingLayer pool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();


        ConvolutionLayer cnn2 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(64).activation(activation).build();

        ConvolutionLayer cnn2_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(64).activation(activation).build();


        SubsamplingLayer pool2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();

        ConvolutionLayer cnn3 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(128).activation(activation).build();

        ConvolutionLayer cnn3_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(128).activation(activation).build();


        SubsamplingLayer pool3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();


        DenseLayer denseLayer = new DenseLayer.Builder().activation(activation)
                .nOut(512).build();

        DenseLayer denseLayer2 = new DenseLayer.Builder().activation(activation)
                .nOut(256).build();

        DenseLayer denseLayer3 = new DenseLayer.Builder().activation(activation)
                .nOut(128).build();


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
                .layer(i++, denseLayer)
                .layer(i++, denseLayer2)
                .layer(i++, denseLayer3)
                .layer(i++, outputLayer)
                .setInputType(InputType.convolutional(h, w, channels))
                .build();

        network = new MultiLayerNetwork(configuration);
        network.init();
    }
```
### 4. 其他

batch_size=100