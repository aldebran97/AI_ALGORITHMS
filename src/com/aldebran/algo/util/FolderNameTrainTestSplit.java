package com.aldebran.algo.util;

import lombok.Getter;
import lombok.Setter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 训练集-测试集分割器（按照目录名分割为不同类）
 *
 * @author aldebran
 * @since 2021-11-18
 */
public class FolderNameTrainTestSplit {

    @Getter
    private File srcDir;

    @Getter
    private File dstDir;

    @Getter
    private int w;

    @Getter
    private int h;

    @Getter
    private File testDir;

    @Getter
    private File trainDir;

    @Getter
    private List<String> labels = new ArrayList<>();

    @Getter
    private double trainRate;

    @Getter
    private double testRate;

    @Getter
    @Setter
    private int maxTrain;

    @Getter
    @Setter
    private int maxTest;


    private static Random defaultRandom = new Random(1);

    public FolderNameTrainTestSplit(File srcDir, File dstDir, int w, int h, double trainRate) throws IOException {
        FileUtil.deleteTrashFile(srcDir);
        FileUtil.deleteTrashFile(dstDir);

        this.srcDir = srcDir;
        this.dstDir = dstDir;
        this.w = w;
        this.h = h;

        this.trainRate = trainRate;
        this.testRate = 1 - trainRate;

        this.trainDir = new File(dstDir, "train");
        this.testDir = new File(dstDir, "test");
        for (File file : srcDir.listFiles()) {
            String name = file.getName();
            if (!name.startsWith(".") && file.isDirectory()) {
                labels.add(name);
            }
        }
    }


    public void convert() throws IOException {

        // 木桶原理，保证样本数量均匀
        int minExamplesNum = Integer.MAX_VALUE;
        int maxExampleNum = 0;
        for (String label : labels) {
            File labelFolder = new File(srcDir, label);
            int c = labelFolder.listFiles().length;
            if (c < minExamplesNum) {
                minExamplesNum = c;
            }
            if (c > maxExampleNum) {
                maxExampleNum = c;
            }
        }

        if (maxTrain == 0) {
            maxTrain = (int) (minExamplesNum * trainRate);
            maxTest = minExamplesNum - maxTrain;
        }

        System.out.println("minExamplesNum: " + minExamplesNum);
        System.out.println("maxExampleNum: " + maxExampleNum);
        System.out.println("max train: " + maxTrain);
        System.out.println("max test: " + maxTest);


        for (String label : labels) {
            File labelFolder = new File(srcDir, label);

            int train = 0, test = 0;
            for (File img : labelFolder.listFiles()) {
                if (train >= maxTrain && test >= maxTest) {
                    break;
                }
                String name = img.getName();
                int dotIndex = name.lastIndexOf(".");
                String outName = name.substring(0, dotIndex) + "_.jpg";
                File outFile = null;
                if (defaultRandom.nextDouble() < trainRate) {
                    if (train < maxTrain) {
                        outFile = new File(trainDir, label);
                        outFile = new File(outFile, outName);
                        train++;
                    } else if (test < maxTest) {
                        outFile = new File(testDir, label);
                        outFile = new File(outFile, outName);
                        test++;
                    }
                } else {
                    if (test < maxTest) {
                        outFile = new File(testDir, label);
                        outFile = new File(outFile, outName);
                        test++;
                    } else if (train < maxTrain) {
                        outFile = new File(trainDir, label);
                        outFile = new File(outFile, outName);
                        train++;
                    }
                }

                if (outFile != null) {
                    ImageUtil.resize(img, w, h, outFile);
                }
            }
            System.out.println("train: " + train);
            System.out.println("test: " + test);
        }
    }
}
