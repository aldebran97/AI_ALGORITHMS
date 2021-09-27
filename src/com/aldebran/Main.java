package com.aldebran;

import com.aldebran.algo.Vector;
import com.aldebran.algo.cnn.PictureClassification;
import com.aldebran.algo.k_mean.KMean;
import com.aldebran.algo.util.CIFAR10Converter;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Main {

    private static String datasetBaseDir = "/Users/aldebran/Downloads/dataset/";

    public static void main(String[] args) throws Exception {

        // 设置可用内存大小
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", 60 * 1024 * 1024 * 1024L + "");
        System.setProperty("org.bytedeco.javacpp.maxbytes", 60 * 1024 * 1024 * 1024L + "");

//        testKMean(); // 测试K-Mean

//        testDatasetConvert(); // 测试CIFAR10 数据集转化

        testCNNPictureClassification(); // 测试卷积神经网络图片分类

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
                new File(datasetBaseDir + "cifar10-dataset-all"), 6000
        );
        cifar10Converter.convert();
        cifar10Converter.split(0.8,
                new File(datasetBaseDir + "cifar10/train"),
                new File(datasetBaseDir + "cifar10/test"));
    }

    public static void testCNNPictureClassification() throws Exception {
        PictureClassification classification = new PictureClassification(
                32, 32, 3,
                new File(datasetBaseDir + "cifar10/train"),
                new File(datasetBaseDir + "cifar10/test"));
        classification.setBatchSize(100);

        classification.loadData();
        System.out.println("after loading data");
//        classification.buildNetwork();
        classification.buildVGG16();
        System.out.println("after build network");
        classification.train(50);
        System.out.println("after train");
        System.out.println(classification.evaluate().stats(true));
        System.out.println("after test");
    }


}
