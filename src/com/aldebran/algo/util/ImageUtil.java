package com.aldebran.algo.util;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Stack;

/**
 * 图片工具类
 *
 * @author aldebran
 */
public class ImageUtil {

    // 缩放图片尺寸
    public static BufferedImage resize(BufferedImage bufferedImage, int width, int height) {
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics g = result.createGraphics();
        g.drawImage(bufferedImage, 0, 0, width, height, null);
        return result;
    }

    // 成比例缩放
    public static BufferedImage resize(BufferedImage bufferedImage, double percent) {
        int nW = (int) (bufferedImage.getWidth() * percent);
        int nH = (int) (bufferedImage.getHeight() * percent);
        return resize(bufferedImage, nW, nH);
    }


    public static void resize(InputStream inputStream, int w, int h, OutputStream outputStream,
                              String format) throws IOException {
        BufferedImage rImage = ImageIO.read(inputStream);
        BufferedImage wImage = resize(rImage, w, h);
        ImageIO.write(wImage, format, outputStream);
        inputStream.close();
        outputStream.close();
    }

    // 文件级
    public static void resize(File in, int w, int h, File out) throws IOException {
        File p = out.getParentFile();
        if (!p.exists() && !p.mkdirs()) {
            throw new IOException("fail to create dir: " + p.getAbsolutePath());
        }
        String outName = out.getName();
        String extension = outName.substring(outName.lastIndexOf(".") + 1);
        resize(new BufferedInputStream(new FileInputStream(in)),
                w, h, new BufferedOutputStream(new FileOutputStream(out)), extension);
    }

    // 求图片的平均dimension
    public static double[] averageSize(File folder) throws IOException {
        Stack<File> stack = new Stack<>();
        stack.push(folder);
        int count = 0;
        long w = 0;
        long h = 0;
        while (!stack.empty()) {
            File file = stack.pop();
            if (file.getName().startsWith(".")) {
                continue;
            }
            if (file.isFile()) {
//                System.out.println(file);
                BufferedImage bufferedImage = ImageIO.read(file);
                if (bufferedImage == null) {
                    continue;
                }
                w += bufferedImage.getWidth();
                h += bufferedImage.getHeight();
                count++;
            } else {
                for (File sub : file.listFiles()) {
                    stack.push(sub);
                }
            }
        }
        return new double[]{w * 1.0 / count, h * 1.0 / count};
    }


}
