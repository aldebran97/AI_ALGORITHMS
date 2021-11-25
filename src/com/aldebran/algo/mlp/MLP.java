package com.aldebran.algo.mlp;

import com.aldebran.algo.iter.CSVDataSetIterator;
import com.aldebran.algo.util.FileUtil;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Path;

/**
 * 多层感知机
 *
 * @author aldebran
 * @since 2021-11-23
 */
public class MLP implements Serializable {

    @Getter
    private int inputSize;

    @Getter
    private int outputSize;

    private transient CSVDataSetIterator trainIterator;

    private transient CSVDataSetIterator testIterator;

    private MultiLayerNetwork network;

    @Setter
    @Getter
    private int seed;

    // 已训练的周期数
    @Getter
    private int trainEpoch = 0;

    // 内存映射文件
    @Getter
    private transient File mMapFile;

    private transient MemoryWorkspace ws;

    @Getter
    private File modelDir;

    // 保存间隔周期数（每多少周期保存一次）
    @Getter
    @Setter
    private int saveInterval = DEFAULT_SAVE_INTERVAL;

    public static final int DEFAULT_SAVE_INTERVAL = 10;

    public MLP(int inputSize, int outputSize, CSVDataSetIterator trainIterator,
               CSVDataSetIterator testIterator, File modelDir, int seed) throws IOException {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.trainIterator = trainIterator;
        this.testIterator = testIterator;
        this.seed = seed;
        setModelDir(modelDir);
    }

    public MLP(int inputSize, int outputSize, File trainFile,
               File testFile, File modelDir, Charset charset, int batchSize, int seed) throws IOException {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.seed = seed;
        setModelDir(modelDir);
        this.trainIterator = new CSVDataSetIterator(trainFile, charset, batchSize, inputSize, outputSize);
        this.testIterator = new CSVDataSetIterator(testFile, charset, batchSize, inputSize, outputSize);
    }


    public void setModelDir(File modelDir) throws IOException {
        if (modelDir != null) {
            FileUtil.createDir(modelDir);
            this.modelDir = modelDir;
        }
    }

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


    // 激活内存映射文件
    public void activateMMapFile(File f, long size) {
        System.out.println("generate memory map file...");
        if (size != f.length()) {
            f.delete();
        }
        mMapFile = f;
        Path path = f.toPath();
        WorkspaceConfiguration mmap = WorkspaceConfiguration.builder()
                .initialSize(size)
                .policyLocation(LocationPolicy.MMAP)
                .policyLearning(LearningPolicy.NONE)
                .tempFilePath(path.toAbsolutePath().toString())
                .build();

        ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M2").notifyScopeLeft();
    }

    // 关闭工作区
    public void deActiveMMapFile() {
        if (ws != null) {
            ws.notifyScopeLeft();
            ws.close();
            ws.destroyWorkspace();
            ws = null;
        }
    }

    // 训练
    public void train(int epochs) throws Exception {
        for (int i = 0; i < epochs; i++) {
            System.out.println("epoch: " + (trainEpoch + 1));
            network.fit(trainIterator);
            trainIterator.reset();
            System.out.println(regressionEvaluate().stats());
//            System.out.println("memory: " + network.memoryInfo(batchSize, InputType.convolutional(h, w, channels)));
            System.out.println();
            if (trainEpoch != 0 && trainEpoch % saveInterval == saveInterval - 1) {
                save();
            }
            trainEpoch++;
        }
    }

    // 测试
    public RegressionEvaluation regressionEvaluate() {

//        System.out.println("test");
        testIterator.reset();
        RegressionEvaluation eval = new RegressionEvaluation(outputSize);
        while (testIterator.hasNext()) {
            DataSet t = testIterator.next();
            INDArray features = t.getFeatures();
            INDArray lables = t.getLabels();
            INDArray predicted = network.output(features, false);

//            System.out.println("labels: " + lables);
//            System.out.println("predicted: " + predicted);

            eval.eval(lables, predicted);


        }

        return eval;
    }

    public double[] predict(double[] input) {
        deActiveMMapFile();
        INDArray in = Nd4j.zeros(new int[]{1, inputSize});
        for (int i = 0; i < inputSize; i++) {
            in.putScalar(new int[]{0, i}, input[i]);
        }
//        System.out.println(in);
        INDArray predicted = network.output(in, false);
        double[] result = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            result[i] = predicted.getDouble(new int[]{0, i});
        }
//        System.out.println(predicted);
        return result;
    }

    /**
     * 保存模型和配置
     *
     * @throws IOException
     */
    public void save() throws IOException {
        if (modelDir == null) {
            return;
        }
        // 保存模型
        File modelFile = new File(modelDir, "" + trainEpoch);
        FileUtil.createFile(modelFile);
        ModelSerializer.writeModel(network, modelFile, true);
        // 保存基本数据
        File descFile = new File(modelDir, "desc");
        FileUtil.createFile(descFile);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(descFile));
        objectOutputStream.writeObject(this);
        objectOutputStream.close();
    }

}
