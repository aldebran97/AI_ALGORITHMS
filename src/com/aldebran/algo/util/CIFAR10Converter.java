package com.aldebran.algo.util;

import lombok.Getter;
import lombok.Setter;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * CIFAR-10数据集转换器
 *
 * @author aldebran
 */
public class CIFAR10Converter {

    @Getter
    private File srcDir;

    @Getter
    private File desDir;

    @Getter
    private File trainDir;

    @Getter
    private File testDir;

    @Getter
    private double trainRate;

    @Getter
    private int numEachClass;

    private static Random defaultRandom = new Random(1);

    @Getter
    @Setter
    private Random random = defaultRandom;

    private List<String> indexLabelList = new ArrayList<>();

    public CIFAR10Converter(File srcDir, File desDir, double trainRate, int numEachClass) {
        this.srcDir = srcDir;
        this.desDir = desDir;
        this.trainRate = trainRate;
        this.numEachClass = numEachClass;
        this.trainDir = new File(desDir, "train");
        this.testDir = new File(desDir, "test");
    }

    private void readLabels() throws IOException {
        File labelDescFile = new File(srcDir, "batches.meta.txt");
        InputStream inputStream = new BufferedInputStream(new FileInputStream(labelDescFile));
        int data;
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        while ((data = inputStream.read()) != -1) {
            byteArrayOutputStream.write(data);
        }
        byteArrayOutputStream.close();
        inputStream.close();
        String str = new String(byteArrayOutputStream.toByteArray());
        str = str.replaceAll("\\s+", "\n");

        for (String s : str.split("\n")) {
            if (!s.isEmpty()) {
                indexLabelList.add(s);
            }
        }
    }

    public List<String> getLabels() {
        return indexLabelList;
    }

    public void convert() throws IOException {
        readLabels(); // 读取标签名
        int trainNum = (int) (trainRate * numEachClass);
        int testNum = numEachClass - trainNum;
        for (String label : indexLabelList) {
            File trainLabelFolder = new File(trainDir, label);
            FileUtil.createDir(trainLabelFolder);
            File testLabelFolder = new File(testDir, label);
            FileUtil.createDir(testLabelFolder);
        }

        for (File sub : srcDir.listFiles()) {
            String subName = sub.getName();
            if (subName.startsWith("data_batch") && subName.endsWith(".bin")) {
                InputStream inputStream = new BufferedInputStream(new FileInputStream(sub));
                byte[] imageContent = new byte[1024 * 3 + 1];
                int writeIndex = 0;
                int pictureIndex = 0; // 图片索引
                while (writeIndex < imageContent.length) {
                    int data = inputStream.read();
                    if (data == -1) {
                        break;
                    }
                    imageContent[writeIndex++] = (byte) data;
                    if (writeIndex == imageContent.length) {
                        String labelName = indexLabelList.get(imageContent[0]);
                        File trainLabelFolder = new File(trainDir, labelName);
                        File testLabelFolder = new File(testDir, labelName);
                        double r = random.nextDouble();
                        if (r < trainRate) {
                            if (trainLabelFolder.listFiles().length < trainNum) {
                                File outFile = new File(trainLabelFolder, "" + pictureIndex + ".jpg");
                                FileUtil.createFile(outFile);
                                OutputStream fileOut = new BufferedOutputStream(new FileOutputStream(outFile));
                                ImageIO.write(getBufferedImage(imageContent), "jpeg", fileOut);
                                fileOut.close();
                            }
                        } else {
                            if (testLabelFolder.listFiles().length < testNum) {
                                File outFile = new File(testLabelFolder, "" + pictureIndex + ".jpg");
                                FileUtil.createFile(outFile);
                                OutputStream fileOut = new BufferedOutputStream(new FileOutputStream(outFile));
                                ImageIO.write(getBufferedImage(imageContent), "jpeg", fileOut);
                                fileOut.close();
                            }
                        }
                        // 重用
                        pictureIndex++;
                        writeIndex = 0;
                    }
                }
            }
        }
    }

    private BufferedImage getBufferedImage(byte[] imageContent) {
        // 图片转换
        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        for (int row = 0; row < 32; row++) {
            for (int col = 0; col < 32; col++) {
                Color color = new Color(
                        imageContent[1 + 1024 * 0 + row * 32 + col] & 0xFF,
                        imageContent[1 + 1024 * 1 + row * 32 + col] & 0xFF,
                        imageContent[1 + 1024 * 2 + row * 32 + col] & 0xFF);
                image.setRGB(col, row, color.getRGB());
            }
        }
        return image;
    }
}
