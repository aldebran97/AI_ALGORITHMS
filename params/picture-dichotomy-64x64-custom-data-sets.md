### 情况

```text
训练集大小 350 测试集大小 118 <br>
分类数2 像素64x64 <br>
芯片apple m1 <br>
系统 macos big sur 11.6
```


### 100周期结果

```text
epoch: 100


========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        1.0000
 Precision:       1.0000
 Recall:          1.0000
 F1 Score:        1.0000
Precision, recall & F1: reported for positive class (class 1 - "other") only


=========================Confusion Matrix=========================
  0  1
-------
 59  0 | 0 = classXXX
  0 59 | 1 = others

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
```

### 网络结构

```java
public void buildNetwork() {

        Activation activation = Activation.RELU;
        LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

        ConvolutionLayer cnn1 = new ConvolutionLayer.Builder(3, 3).nIn(channels).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(32).activation(activation).build();

//        ConvolutionLayer cnn1_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
//                .convolutionMode(ConvolutionMode.Same)
//                .nOut(32).activation(activation).build();

        SubsamplingLayer pool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();


        ConvolutionLayer cnn2 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(64).activation(activation).build();

//        ConvolutionLayer cnn2_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
//                .convolutionMode(ConvolutionMode.Same)
//                .nOut(64).activation(activation).build();


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


        LSTM lstm = new LSTM.Builder().activation(activation)
                .nOut(512).build();
//
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
//                .layer(i++, cnn1_1)
                .layer(i++, pool1)
                .layer(i++, cnn2)
//                .layer(i++, cnn2_1)
                .layer(i++, pool2)
                .layer(i++, cnn3)
                .layer(i++, cnn3_1)
                .layer(i++, pool3)
                .layer(i++, lstm)
                .layer(i++, lstm2)
//                .layer(i++, denseLayer)
                .layer(i++, outputLayer)
                .setInputType(InputType.convolutional(h, w, channels))
                .build();
        configuration.setTrainingWorkspaceMode(WorkspaceMode.ENABLED);
        configuration.setInferenceWorkspaceMode(WorkspaceMode.NONE);


        network = new MultiLayerNetwork(configuration);
        network.init();
    }
```
### 初始化

```java
PictureClassification pictureClassification = new PictureClassification(
        w, h, 3, 50, 1, 
        trainDir, testDir, modelDir);
```
