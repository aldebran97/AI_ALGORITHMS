package com.aldebran.algo.cnn;

import com.aldebran.algo.util.FileUtil;
import com.aldebran.algo.util.ImageUtil;
import lombok.Getter;
import lombok.Setter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Path;
import java.util.*;

/**
 * CNN图片分类
 *
 * @author aldebran
 * @since 2021-09-27
 */
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

    public static int DEFAULT_BATCH_SIZE = 50;

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
    private int trainEpoch = 0;

    // 内存映射文件
    @Getter
    private transient File mMapFile;

    private transient MemoryWorkspace ws;

    private List<String> labels;

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

        if (labels == null) {
            labels = trainDataSetIterator.getLabels();
        }

        // test
        FileSplit fileSplitTest = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, random);
        ImageRecordReader imageRecordReaderTest = new ImageRecordReader(h, w, channels, labelGenerator);
        imageRecordReaderTest.initialize(fileSplitTest);
        this.testDataSetIterator = new RecordReaderDataSetIterator(
                imageRecordReaderTest, batchSize, 1, nLabels);
        DataNormalization dataNormalizationTest = new ImagePreProcessingScaler();
        dataNormalizationTest.fit(testDataSetIterator);
        testDataSetIterator.setPreProcessor(dataNormalizationTest);

        if (!labels.equals(testDataSetIterator.getLabels())) {
            throw new RuntimeException("error labels");
        }
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
        System.out.println("nLabels: " + nLabels);
        for (int i = 0; i < epochs; i++) {
            System.out.println("epoch: " + (trainEpoch + 1));
            network.fit(trainDataSetIterator);
            trainDataSetIterator.reset();
            System.out.println(evaluate().stats(true));
//            System.out.println("memory: " + network.memoryInfo(batchSize, InputType.convolutional(h, w, channels)));
            System.out.println();
            if (trainEpoch != 0 && trainEpoch % saveInterval == saveInterval - 1) {
                save();
            }
            trainEpoch++;
        }
    }


    public Evaluation evaluate() {
        Evaluation evaluation = network.evaluate(testDataSetIterator);
        testDataSetIterator.reset();
        return evaluation;
    }

    // 预测方法
    public String predict(InputStream inputStream) throws IOException {

        try {
            deActiveMMapFile();
            BufferedImage bufferedImage = ImageIO.read(inputStream);
            if (bufferedImage.getWidth() != w || bufferedImage.getHeight() != h) {
                bufferedImage = ImageUtil.resize(bufferedImage, w, h);
            }
            NativeImageLoader loader = new NativeImageLoader(h, w, channels);
            INDArray input = loader.asMatrix(bufferedImage);
//            System.out.println(Arrays.toString(input.shape()));
            DataNormalization scalar = new ImagePreProcessingScaler(0, 1);
            scalar.transform(input);
            int[] result = network.predict(input);
//            System.out.println(Arrays.toString(result));
//            System.out.println(network.output(input));
            return labels.get(result[0]);
        } finally {
            inputStream.close();
        }
    }

    public String predict(File file) throws IOException {
        return predict(new BufferedInputStream(new FileInputStream(file)));
    }

    public List<String> predict(Iterator<InputStream> inputStreams) throws IOException {
        List<String> result = new ArrayList<>();
        while (inputStreams.hasNext()) {
            result.add(predict(inputStreams.next()));
        }
        return result;
    }

    public List<String> predictFiles(List<File> files) throws IOException {
        Iterator<InputStream> inputStreams = new Iterator<InputStream>() {

            private int i = 0;

            @Override
            public boolean hasNext() {
                return i < files.size();
            }

            @Override
            public InputStream next() {
                File f = files.get(i++);
                try {
                    return new BufferedInputStream(new FileInputStream(f));
                } catch (FileNotFoundException e) {
                    throw new RuntimeException(e);
                }
            }
        };
        return predict(inputStreams);
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

    /**
     * 加载配置和模型
     *
     * @param modelDir   目录
     * @param trainEpoch 训练周期数，索引从0开始
     * @return
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public static PictureClassification loadConfigAndModel(File modelDir, int trainEpoch) throws IOException, ClassNotFoundException {
        File descFile = new File(modelDir, "desc");
        ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(descFile));
        PictureClassification pictureClassification = (PictureClassification) objectInputStream.readObject();
        pictureClassification.network = ModelSerializer.
                restoreMultiLayerNetwork(
                        new File(modelDir, "" + trainEpoch));
        pictureClassification.setModelDir(modelDir);
        return pictureClassification;
    }


}
