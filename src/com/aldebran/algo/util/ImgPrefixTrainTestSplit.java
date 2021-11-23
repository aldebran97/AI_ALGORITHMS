package com.aldebran.algo.util;

import lombok.Getter;
import lombok.Setter;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * 训练集-测试集分割器（按照文件名前缀分割为不同类）
 *
 * @author aldebran
 * @since 2021-11-18
 */
public class ImgPrefixTrainTestSplit {

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

    private Map<String, Integer> labelCount = new HashMap<>();

    private Map<String, Integer> labelCurrentTrain = new HashMap<>();

    private Map<String, Integer> labelCurrentTest = new HashMap<>();

    private Map<String, Integer> labelMaxTrain = new HashMap<>();

    private Map<String, Integer> labelMaxTest = new HashMap<>();

    // 获取标签
    public Set<String> getLabels() {
        return labelCount.keySet();
    }

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

    private String split;

    private static Random defaultRandom = new Random(1);

    public ImgPrefixTrainTestSplit(File srcDir, File dstDir, String split,
                                   int w, int h, double trainRate) throws IOException {
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

        this.split = split;

        for (File file : srcDir.listFiles()) {
            String name = file.getName();
            if (!name.startsWith(".") && file.isFile()) {
                String label = name.substring(0, name.indexOf(split));
                if (!labelCount.containsKey(label)) {
                    labelCount.put(label, 1);
                } else {
                    labelCount.put(label, labelCount.get(label) + 1);
                }
            }
        }

    }

    public void convert() throws IOException {

        // 一些统计
        for (String label : labelCount.keySet()) {
            int count = labelCount.get(label);
            if (maxTrain == 0) {
                int maxTrain = (int) (trainRate * count);
                int maxTest = count - maxTrain;
                labelMaxTrain.put(label, maxTrain);
                labelMaxTest.put(label, maxTest);
            } else {
                labelMaxTrain.put(label, maxTrain);
                labelMaxTest.put(label, maxTest);
            }
            labelCurrentTest.put(label, 0);
            labelCurrentTrain.put(label, 0);
        }
        System.out.println("labelCount: " + labelCount);
        System.out.println("labelMaxTrain: " + labelMaxTrain);
        System.out.println("labelMaxTest: " + labelMaxTest);

        // 分配 转换
        for (File file : srcDir.listFiles()) {
            String name = file.getName();
            int dotIndex = name.lastIndexOf(".");
            String outName = name.substring(0, dotIndex) + "_.jpg";
            if (!name.startsWith(".") && file.isFile()) {
                String label = name.substring(0, name.indexOf(split));
                File outFile = null;
                if (defaultRandom.nextDouble() < trainRate) {
                    if (labelCurrentTrain.get(label) < labelMaxTrain.get(label)) {
                        outFile = new File(trainDir, label);
                        outFile = new File(outFile, outName);
                        labelCurrentTrain.put(label, labelCurrentTrain.get(label) + 1);
                    } else if (labelCurrentTest.get(label) < labelMaxTest.get(label)) {
                        outFile = new File(testDir, label);
                        outFile = new File(outFile, outName);
                        labelCurrentTest.put(label, labelCurrentTest.get(label) + 1);
                    }
                } else {
                    if (labelCurrentTest.get(label) < labelMaxTest.get(label)) {
                        outFile = new File(testDir, label);
                        outFile = new File(outFile, outName);
                        labelCurrentTest.put(label, labelCurrentTest.get(label) + 1);
                    } else if (labelCurrentTrain.get(label) < labelMaxTrain.get(label)) {
                        outFile = new File(trainDir, label);
                        outFile = new File(outFile, outName);
                        labelCurrentTrain.put(label, labelCurrentTrain.get(label) + 1);
                    }
                }
                if (outFile != null) {
                    ImageUtil.resize(file, w, h, outFile);
                }

            }
        }

        System.out.println("labelCurrentTrain: " + labelCurrentTrain);
        System.out.println("labelCurrentTest: " + labelCurrentTest);

    }
}
