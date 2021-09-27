package com.aldebran.algo.util;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.CopyOption;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.rmi.RemoteException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CIFAR-10数据集转换器
 * @author aldebran
 */
public class CIFAR10Converter {

    private File inDir;

    private File outDir;

    private int countEachClassification;

    private List<String> indexLabelList = new ArrayList<>();

    public CIFAR10Converter(File inDir, File outDir, int countEachClassification) {
        this.inDir = inDir;
        this.outDir = outDir;
        this.countEachClassification = countEachClassification;
    }

    // bin -> 图片文件
    public void convert() throws IOException {
        readLabels(); // 读取标签名
        for (String label : indexLabelList) {
            File labelFolder = new File(outDir, label);
            FileUtil.createDir(labelFolder);
        }
        for (File sub : inDir.listFiles()) {
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
                        File labelFolder = new File(outDir, labelName);
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
                        // 图片保存
                        File outFile = new File(labelFolder, "" + pictureIndex + ".jpg");
                        pictureIndex++;
                        FileUtil.createFile(outFile);
                        OutputStream fileOut = new BufferedOutputStream(new FileOutputStream(outFile));
                        ImageIO.write(image, "jpeg", fileOut);
                        fileOut.close();
                        // 重用
                        writeIndex = 0;
                    }
                }
            }
        }

        // 控制数量
        for (File labelFolder : Objects.requireNonNull(outDir.listFiles())) {
            if (labelFolder.getName().startsWith(".")) {
                if (!labelFolder.delete()) {
                    throw new RuntimeException("fail to delete file: " + labelFolder.getAbsolutePath());
                }
                continue;
            }
            int c = 0;
            for (File sub : Objects.requireNonNull(labelFolder.listFiles())) {
                if (sub.getName().startsWith(".")) {
                    if (!sub.delete()) {
                        throw new RuntimeException("fail to delete file: " + sub.getAbsolutePath());
                    }
                    continue;
                }
                c++;
                if (c > countEachClassification) {
                    if (!sub.delete()) {
                        throw new RuntimeException("fail to delete file: " + sub.getAbsolutePath());
                    }
                }
            }
        }

    }

    // 分割训练集和测试集
    public void split(double trainingRate, File trainDir, File testDir) throws IOException {
        FileUtil.createDir(trainDir);
        FileUtil.createDir(testDir);
        for (File labelFolder : outDir.listFiles()) {
            String labelName = labelFolder.getName();
            if (labelName.startsWith(".")) {
                FileUtil.deleteFile(labelFolder);
                continue;
            }
            File outTrainDir = new File(trainDir, labelFolder.getName());
            File outTestDir = new File(testDir, labelFolder.getName());
            FileUtil.createDir(outTrainDir);
            FileUtil.createDir(outTestDir);
            int trainCount = 0;
            int trainMax = (int) (labelFolder.listFiles().length * trainingRate);
            int testCount = 0;
            int testMax = labelFolder.listFiles().length - trainMax;
            for (File sub : labelFolder.listFiles()) {
                if (sub.getName().startsWith(".")) {
                    FileUtil.deleteFile(sub);
                    continue;
                }
                if (trainCount < trainMax) {
                    File newFile = new File(outTrainDir, sub.getName());
                    FileUtil.createFile(newFile);
                    Files.copy(sub.toPath(), newFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    trainCount++;
                } else if (testCount < testMax) {
                    File newFile = new File(outTestDir, sub.getName());
                    FileUtil.createFile(newFile);
                    Files.copy(sub.toPath(), newFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    testCount++;
                } else {
                    break;
                }
            }
        }
    }

    private void readLabels() throws IOException {
        File labelDescFile = new File(inDir, "batches.meta.txt");
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
}
