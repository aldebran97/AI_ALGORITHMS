package com.aldebran.algo.cnn;

import com.aldebran.algo.util.FileUtil;
import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import lombok.Getter;
import lombok.Setter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.Random;

public class PictureClassification implements Serializable {

    // 图片宽度，构造指定后不可再修改
    @Getter
    private final int w;

    // 图片高度，构造指定后不可再修改
    @Getter
    private final int h;

    // 图片深度，构造指定后不可再修改
    @Getter
    private final int channels;

    // 标签数，加载数据后不可再修改
    @Getter
    private int nLabels = -1;

    // 批大小，构造指定后不可再修改
    @Getter
    private final int batchSize;

    public static int DEFAULT_BATCH_SIZE = 100;

    // 随机相关，构造指定后不可再修改
    private final int seed;

    public static final int DEFAULT_SEED = 1;

    private final Random random;

    // 训练目录，setter额外处理
    @Getter
    private File trainDir;

    // 测试目录，setter额外处理
    @Getter
    private File testDir;

    private transient DataSetIterator trainDataSetIterator;

    private transient DataSetIterator testDataSetIterator;

    // 网络结构
    private transient MultiLayerNetwork network;

    // 保存模型和描述的目录,如果没有就不保存
    @Getter
    private File modelDir;

    // 保存间隔周期数（每多少周期保存一次）
    @Getter
    @Setter
    private int saveInterval = DEFAULT_SAVE_INTERVAL;

    public static final int DEFAULT_SAVE_INTERVAL = 10;

    // 已训练的周期数
    @Getter
    private int trainEpochs = 0;

    public PictureClassification(int w, int h, int channels, int batchSize, int seed,
                                 File trainDir, File testDir, File modelDir) throws IOException {
        this(w, h, channels, batchSize, seed);
        setTrainDir(trainDir);
        setTestDir(testDir);
        setModelDir(modelDir);
    }

    public PictureClassification(int w, int h, int channels, int batchSize, int seed) {
        this.w = w;
        this.h = h;
        this.channels = channels;
        this.batchSize = batchSize;
        this.seed = seed;
        this.random = new Random(seed);
    }

    // 删除临时文件
    private void deleteUselessFiles(File folder) {
        if (folder.isFile()) {
            String name = folder.getName();
            if (name.startsWith(".")) {
                if (!folder.delete()) {
                    throw new RuntimeException("fail to delete file: " + folder.getAbsolutePath());
                }
            }
        } else {
            File[] subs = folder.listFiles();
            if (subs != null) {
                for (File sub : subs) {
                    deleteUselessFiles(sub);
                }
            }
        }
    }

    public void setModelDir(File modelDir) throws IOException {
        FileUtil.createDir(modelDir);
        this.modelDir = modelDir;
    }

    // 设置训练数据集
    public void setTrainDir(File trainDir) {
        deleteUselessFiles(trainDir);
        this.trainDir = trainDir;
        if (nLabels == -1) {
            nLabels = trainDir.listFiles().length;
        } else if (trainDir.listFiles().length != nLabels) {
            throw new RuntimeException("error labels number");
        }
    }

    // 设置测试数据集
    public void setTestDir(File testDir) {
        deleteUselessFiles(testDir);
        this.testDir = testDir;
        if (nLabels == -1) {
            nLabels = trainDir.listFiles().length;
        } else if (trainDir.listFiles().length != nLabels) {
            throw new RuntimeException("error labels number");
        }
    }

    // 加载数据
    public void loadData() throws IOException {
        // train
        FileSplit fileSplitTrain = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, random);
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();

        ImageRecordReader imageRecordReaderTrain = new ImageRecordReader(h, w, channels, labelGenerator);
        imageRecordReaderTrain.initialize(fileSplitTrain);

        this.trainDataSetIterator = new RecordReaderDataSetIterator(
                imageRecordReaderTrain, batchSize, 1, nLabels);
        DataNormalization dataNormalizationTrain = new ImagePreProcessingScaler();
        dataNormalizationTrain.fit(trainDataSetIterator);
        trainDataSetIterator.setPreProcessor(dataNormalizationTrain);
        // test
        FileSplit fileSplitTest = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, random);
        ImageRecordReader imageRecordReaderTest = new ImageRecordReader(h, w, channels, labelGenerator);
        imageRecordReaderTest.initialize(fileSplitTest);
        this.testDataSetIterator = new RecordReaderDataSetIterator(
                imageRecordReaderTest, batchSize, 1, nLabels);
        DataNormalization dataNormalizationTest = new ImagePreProcessingScaler();
        dataNormalizationTest.fit(testDataSetIterator);
        testDataSetIterator.setPreProcessor(dataNormalizationTest);

//        int c = 0;
//        while (testDataSetIterator.hasNext()) {
//            DataSet dataSet = testDataSetIterator.next();
//            System.out.println(dataSet.getFeatureMatrix().length()); //h * w * channels * batchSize
//            c++;
//        }
//        System.out.println("c: " + c); // 图片数/批大小向上取整


    }

    // 创建网络结构
    public void buildNetwork() {

        Activation activation = Activation.RELU;
        LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

        ConvolutionLayer cnn1 = new ConvolutionLayer.Builder(3, 3).nIn(channels).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(128).activation(activation).build();

        ConvolutionLayer cnn1_2 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(128).activation(activation).build();

        SubsamplingLayer pool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();


        ConvolutionLayer cnn2 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(256).activation(activation).build();

        ConvolutionLayer cnn2_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(256).activation(activation).build();


        SubsamplingLayer pool2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();

        ConvolutionLayer cnn3 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(512).activation(activation).build();

        ConvolutionLayer cnn3_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(512).activation(activation).build();

        SubsamplingLayer pool3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();

        DenseLayer denseLayer = new DenseLayer.Builder().activation(activation)
                .nOut(4096).build();

        DenseLayer denseLayer2 = new DenseLayer.Builder().activation(activation)
                .nOut(1000).build();

        OutputLayer outputLayer = new OutputLayer.Builder(lossFunction)
                .nOut(nLabels).activation(Activation.SOFTMAX).build();

        int i = 0;

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().seed(seed)
                .l2(0.01)
                .updater(Updater.ADAM)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dropOut(0.5)
                .list()
                .layer(i++, cnn1)
                .layer(i++, cnn1_2)
                .layer(i++, pool1)
                .layer(i++, cnn2)
                .layer(i++, cnn2_1)
                .layer(i++, pool2)
                .layer(i++, cnn3)
                .layer(i++, cnn3_1)
                .layer(i++, pool3)
                .layer(i++, denseLayer)
                .layer(i++, denseLayer2)
                .layer(i++, outputLayer)
                .setInputType(InputType.convolutional(h, w, channels))
                .build();

        network = new MultiLayerNetwork(configuration);
        network.init();
    }

    // 训练
    public void train(int epochs) throws Exception {
        for (int i = 0; i < epochs; i++) {
            System.out.println("epoch: " + (trainEpochs + 1));
            network.fit(trainDataSetIterator);
            System.gc();
            Thread.sleep(10 * 1000);
            trainDataSetIterator.reset();
            System.out.println(evaluate().stats(true));
            System.out.println();
            if (trainEpochs != 0 && trainEpochs % saveInterval == 0) {
                save();
            }
            trainEpochs++;
            System.gc();
        }
    }


    public Evaluation evaluate() {
        Evaluation evaluation = network.evaluate(testDataSetIterator);
        testDataSetIterator.reset();
        return evaluation;
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
        File modelFile = new File(modelDir, "" + trainEpochs);
        FileUtil.createFile(modelFile);
        ModelSerializer.writeModel(network, modelFile, true);
        // 保存基本数据
        File descFile = new File(modelDir, "desc");
        FileUtil.createFile(descFile);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(descFile));
        objectOutputStream.writeObject(this);
        objectOutputStream.close();
    }


    /**
     * 加载配置和模型
     *
     * @param modelDir 目录
     * @return
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public static PictureClassification loadConfigAndModel(File modelDir) throws IOException, ClassNotFoundException {
        File descFile = new File(modelDir, "desc");
        ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(descFile));
        PictureClassification pictureClassification = (PictureClassification) objectInputStream.readObject();
        pictureClassification.network = ModelSerializer.
                restoreMultiLayerNetwork(
                        new File(modelDir, "" + pictureClassification.trainEpochs));
        pictureClassification.setModelDir(modelDir);
        return pictureClassification;
    }
}
