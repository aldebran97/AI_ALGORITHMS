package com.aldebran.runnable;

import com.aldebran.algo.cnn.PictureClassification;

import java.io.File;
import java.io.IOException;

public class TryPictureClassification {

    static String trainDir = "/Users/aldebran/Downloads/dataset/cifar10/train";

    static String testDir = "/Users/aldebran/Downloads/dataset/cifar10/test";

    static String modelDir = "/Users/aldebran/Downloads/dataset/model";

    static String mMapFile = "/Users/aldebran/Downloads/dataset/tmpFile";


    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("javacpp.platform", "linux-arm64");
//        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
        PictureClassification classification = first();

        classification.train(100);

        System.out.println(classification.evaluate().stats(true));

        classification.deActiveMMapFile();

    }

    // 首次训练
    public static PictureClassification first() throws IOException {
        PictureClassification classification = new PictureClassification(
                32, 32, 3, 100, 1,
                new File(trainDir),
                new File(testDir),
                new File(modelDir));
        classification.activateMMapFile(new File(mMapFile), 40L * 1024 * 1024 * 1024);
        classification.setSaveInterval(10);
        classification.loadData();
        classification.buildNetwork();
        return classification;
    }


    // 加载已经训练好的模型
    public static PictureClassification loadFromFile() throws IOException, ClassNotFoundException {
        PictureClassification classification =
                PictureClassification.loadConfigAndModel(new File(modelDir), 99);
        classification.activateMMapFile(new File(mMapFile), 40L * 1024 * 1024 * 1024);
        classification.loadData();
        return classification;
    }

}
