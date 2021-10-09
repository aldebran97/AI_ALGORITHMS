package com.aldebran;

import com.aldebran.algo.Vector;
import com.aldebran.algo.cnn.PictureClassification;
import com.aldebran.algo.k_mean.KMean;
import com.aldebran.algo.util.CIFAR10Converter;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Main {

    private static String datasetBaseDir = "/home/wenjiajun/dataset/";

    private static String mMapFile = "/home/wenjiajun/dataset/tmpFile";

    public static void main(String[] args) throws Exception {

        testCNNPictureClassification(); // 测试卷积神经网络图片分类

//        testKMean(); // 测试K-Mean

//        testDatasetConvert(); // 测试CIFAR10 数据集转化

    }

    public static void testKMean() {
        KMean kMean = new KMean(2, Arrays.asList(
                new Vector(0, 0), new Vector(1, 1), new Vector(0, 1),
                new Vector(3, 3), new Vector(3, 4), new Vector(4, 3)
        ).iterator(), 0.001);

        System.out.println(kMean.getCollections());
    }

    public static void testDatasetConvert() throws IOException {
        CIFAR10Converter cifar10Converter = new CIFAR10Converter(
                new File(datasetBaseDir + "cifar-10-batches-bin"),
                new File(datasetBaseDir + "cifar10"), 0.8, 1000
        );
        cifar10Converter.convert();
    }

    public static void testCNNPictureClassification() throws Exception {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
//        System.setProperty("org.bytedeco.javacpp.maxbytes", "0G");
        PictureClassification classification = new PictureClassification(
                32, 32, 3, 100, 1,
                new File(datasetBaseDir + "cifar10/train"),
                new File(datasetBaseDir + "cifar10/test"),
                new File(datasetBaseDir + "model"));
        classification.activateMMapFile(new File(mMapFile), 10L * 1024 * 1024 * 1024);
        classification.setSaveInterval(10);
        classification.loadData();
        System.out.println("after loading data");
        classification.buildNetwork();
        System.gc();
        System.out.println("after build network");
        classification.train(50);
        System.out.println("after train");
        System.out.println(classification.evaluate().stats(true));
        System.out.println("after test");
    }


}
