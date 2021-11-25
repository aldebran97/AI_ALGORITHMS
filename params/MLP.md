### 情况
模拟 z = 2 * x + y
训练集大小 10000 测试集大小 1000 <br>
输入2 输出1 <br>
芯片apple m1 <br>
系统 macos big sur 11.6

### 499周期结果

```text
epoch: 499
Column    MSE            MAE            RMSE           RSE            PC             R^2            
col_0     9.24652e-03    2.60771e-02    9.61588e-02    -3.02596e-07   1.00000e+00    1.00000e+00   
```

### 网络结构

```java
    public void buildNetwork() {
        int i = 0;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.NORMAL)
                .weightDecay(0.999)
                .l1(0.2)
                .l2(0.2)
                .biasInit(1)
                .list()
                .layer(i++, new DenseLayer.Builder().nIn(inputSize).nOut(16)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(i++, new LSTM.Builder().nOut(16)
                        .activation(Activation.LEAKYRELU)
                        .build())
//                .layer(i++, new LSTM.Builder().nOut(16)
//                        .weightInit(WeightInit.XAVIER)
//                        .activation(Activation.RELU)
//                        .build())
                .layer(i++, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.LEAKYRELU)
                        .nOut(outputSize).build())
                .build();

        network = new MultiLayerNetwork(conf);
        network.init();

    }
```

### 初始化
```text
批大小为50，训练集和测试集随机生成，随机种子取1
```
