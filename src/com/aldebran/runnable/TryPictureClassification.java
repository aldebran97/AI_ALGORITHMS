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
        PictureClassification classification = new PictureClassification(
                32, 32, 3, 100, 1,
                new File(trainDir),
                new File(testDir),
                new File(modelDir));
        classification.activateMMapFile(new File(mMapFile), 40L * 1024 * 1024 * 1024);
        classification.setSaveInterval(10);
        classification.loadData();
        System.out.println("after loading data");
        classification.buildNetwork();
        System.out.println("after build network");
        classification.train(100);
        System.out.println("after train");
        System.out.println(classification.evaluate().stats(true));
        System.out.println("after test");
    }
}
