package com.aldebran;

import com.aldebran.algo.Vector;
import com.aldebran.algo.cnn.PictureClassification;
import com.aldebran.algo.k_mean.KMean;
import com.aldebran.algo.util.CIFAR10Converter;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class Main {

    private static String datasetBaseDir = "C:/Users/hasee/Desktop/tmp/cifar10-dataset-all/";

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
                new File(datasetBaseDir + "cifar10-dataset-all"), 6000
        );
        cifar10Converter.convert();
        cifar10Converter.split(0.8,
                new File(datasetBaseDir + "cifar10/train"),
                new File(datasetBaseDir + "cifar10/test"));
    }

    public static void testCNNPictureClassification() throws Exception {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "3G");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "3G");
        Path path = Paths.get("c:/Users/hasee/Desktop/tmp/tmpFile");
        Files.delete(path);
        WorkspaceConfiguration mmap = WorkspaceConfiguration.builder()
                .initialSize(40 * 1024 * 1024 * 1024L)
                .policyLocation(LocationPolicy.MMAP)
                .tempFilePath(path.toAbsolutePath().toString())
                .build();
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M2")) {
            PictureClassification classification = new PictureClassification(
                    32, 32, 3, 100, 1,
                    new File(datasetBaseDir + "cifar10/train"),
                    new File(datasetBaseDir + "cifar10/test"),
                    new File(datasetBaseDir + "model"));
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
        } finally {
            Files.delete(path);
        }
    }


}
