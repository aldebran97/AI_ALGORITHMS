package com.aldebran.algo.cnn;

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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class PictureClassification {

    private int w;

    private int h;

    private int channels;

    private int nLabels;

    private int batchSize = DEFAULT_BATCH_SIZE;

    private static int DEFAULT_BATCH_SIZE = 54;

    private int seed = DEFAULT_SEED;

    private static int DEFAULT_SEED = 1;

    private Random random;

    private File trainDir;

    private File testDir;

    private DataSetIterator trainDataSetIterator;

    private DataSetIterator testDataSetIterator;

    private MultiLayerNetwork network;

    public PictureClassification(int w, int h, int channels, File trainDir, File testDir) {
        this.w = w;
        this.h = h;
        this.channels = channels;
        setBatchSize(batchSize);
        setSeed(seed);
        this.trainDir = trainDir;
        this.testDir = testDir;
        File[] testDirSubs = testDir.listFiles();
        File[] trainDirSubs = trainDir.listFiles();
        if (testDirSubs != null && trainDirSubs != null) {
            for (File sub : testDirSubs) {
                if (sub.getName().startsWith(".")) {
                    if (!sub.delete()) {
                        throw new RuntimeException("删除文件失败：" + sub.getAbsolutePath());
                    }
                }
            }
            for (File sub : trainDirSubs) {
                if (sub.getName().startsWith(".")) {
                    if (!sub.delete()) {
                        throw new RuntimeException("删除文件失败：" + sub.getAbsolutePath());
                    }
                }
            }
            testDirSubs = testDir.listFiles();
            trainDirSubs = trainDir.listFiles();
            if (testDirSubs.length != trainDirSubs.length) {
                throw new RuntimeException("标签数量不对应");
            }
            this.nLabels = trainDirSubs.length;
        }
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setSeed(int seed) {
        this.seed = seed;
        random = new Random(seed);
    }

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

    public void buildNetwork() {
        ConvolutionLayer cnn1 = new ConvolutionLayer.Builder(3, 3).nIn(channels).stride(1, 1)
                .nOut(20).activation(Activation.RELU).build();

        SubsamplingLayer pool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();


        ConvolutionLayer cnn2 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .nOut(50).activation(Activation.RELU).build();

        DenseLayer denseLayer = new DenseLayer.Builder().activation(Activation.RELU)
                .nOut(500).build();

        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(nLabels).activation(Activation.SOFTMAX).build();

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().seed(seed)
                .l2(0.0001)
                .updater(Updater.ADAM)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, cnn1)
                .layer(1, pool1)
                .layer(2, cnn2)
                .layer(3, pool1)
                .layer(4, denseLayer)
                .layer(5, outputLayer)
                .setInputType(InputType.convolutional(h, w, channels))
                .build();

        network = new MultiLayerNetwork(configuration);
        network.init();
    }

    public void buildVGG16() {
        ConvolutionLayer cnn1 = new ConvolutionLayer.Builder(3, 3).nIn(channels).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(20).activation(Activation.RELU).build();

        ConvolutionLayer cnn1_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(20).activation(Activation.RELU).build();

        SubsamplingLayer pool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();

        ConvolutionLayer cnn2 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(40).activation(Activation.RELU).build();

        ConvolutionLayer cnn2_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(40).activation(Activation.RELU).build();

        SubsamplingLayer pool2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();

        ConvolutionLayer cnn3 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(80).activation(Activation.RELU).build();

        ConvolutionLayer cnn3_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(80).activation(Activation.RELU).build();

        ConvolutionLayer cnn3_2 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(80).activation(Activation.RELU).build();

        SubsamplingLayer pool3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();

        ConvolutionLayer cnn4 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(160).activation(Activation.RELU).build();

        ConvolutionLayer cnn4_1 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(160).activation(Activation.RELU).build();

        ConvolutionLayer cnn4_2 = new ConvolutionLayer.Builder(3, 3).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .nOut(160).activation(Activation.RELU).build();

        SubsamplingLayer pool4 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                .stride(2, 2).build();

        DenseLayer denseLayer = new DenseLayer.Builder().activation(Activation.RELU)
                .nOut(1280).build();

        DenseLayer denseLayer_1 = new DenseLayer.Builder().activation(Activation.RELU)
                .nOut(1280).build();

        DenseLayer denseLayer_2 = new DenseLayer.Builder().activation(Activation.RELU)
                .nOut(640).build();

        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(nLabels).activation(Activation.SOFTMAX).build();

        int i = 0;

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().seed(seed)
                .l2(0.01)
                .lrPolicyDecayRate(0.99)
                .updater(Updater.ADAM)
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
                .layer(i++, cnn3_2)
                .layer(i++, pool3)
                .layer(i++, cnn4)
                .layer(i++, cnn4_1)
                .layer(i++, cnn4_2)
                .layer(i++, pool4)
                .layer(i++, denseLayer)
                .layer(i++, denseLayer_1)
                .layer(i++, denseLayer_2)
                .layer(i++, outputLayer)
                .setInputType(InputType.convolutional(h, w, channels))
                .build();

        network = new MultiLayerNetwork(configuration);
        network.init();
    }

    public void train(int epochs) {
        for (int i = 0; i < epochs; i++) {
            System.out.println("epoch: " + (i + 1));
            network.fit(trainDataSetIterator);
            trainDataSetIterator.reset();
            testDataSetIterator.reset();
            System.out.println(evaluate().stats(true));
            System.out.println();
        }
    }


    public Evaluation evaluate() {
        Evaluation evaluation = network.evaluate(testDataSetIterator);
        testDataSetIterator.reset();
        return evaluation;
    }
}
