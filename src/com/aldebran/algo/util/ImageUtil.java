package com.aldebran.algo.util;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;

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
        String outName = out.getName();
        String extension = outName.substring(outName.lastIndexOf(".") + 1);
        resize(new BufferedInputStream(new FileInputStream(in)),
                w, h, new BufferedOutputStream(new FileOutputStream(out)), extension);
    }


}
