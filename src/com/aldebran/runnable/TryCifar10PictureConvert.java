package com.aldebran.runnable;

import com.aldebran.algo.util.CIFAR10Converter;

import java.io.File;
import java.io.IOException;

public class TryCifar10PictureConvert {

    static String originDir = "/Users/aldebran/Downloads/dataset/cifar-10-batches-bin";

    static String destDir = "/Users/aldebran/Downloads/dataset/cifar10";

    static int numEachClass = 100000;

    static double trainRate = 0.8;

    public static void main(String[] args) throws IOException {
        CIFAR10Converter cifar10Converter = new CIFAR10Converter(
                new File(originDir),
                new File(destDir), trainRate, numEachClass
        );
        cifar10Converter.convert();
    }
}
