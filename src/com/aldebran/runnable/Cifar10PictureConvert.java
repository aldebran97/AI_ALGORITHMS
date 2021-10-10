package com.aldebran.runnable;

import com.aldebran.algo.util.CIFAR10Converter;

import java.io.File;
import java.io.IOException;

public class Cifar10PictureConvert {

    static String originDir = "f:/dataset/cifar-10-batches-bin";

    static String destDir = "f:/dataset/cifar10";

    static int numEachClass = 10000;

    static double trainRate = 0.8;

    public static void main(String[] args) throws IOException {
        CIFAR10Converter cifar10Converter = new CIFAR10Converter(
                new File(originDir),
                new File(destDir), trainRate, numEachClass
        );
        cifar10Converter.convert();
    }
}
