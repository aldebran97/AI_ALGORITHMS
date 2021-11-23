package com.aldebran.algo.util;

import java.io.File;
import java.io.IOException;

/**
 * 文件工具
 *
 * @author aldebran
 */
public class FileUtil {

    public static void createFile(File file) throws IOException {
        File p = file.getParentFile();
        if (p != null) {
            createDir(p);
        }
        if (!file.exists() && !file.createNewFile()) {
            throw new IOException("fail to create file: " + file.getAbsolutePath());
        }
    }

    public static void createDir(File dir) throws IOException {
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("fail to create dir: " + dir.getAbsolutePath());
        }
    }

    public static void deleteFile(File file) throws IOException {
        if (file.exists() && !file.exists()) {
            throw new IOException("fail to delete file: " + file.getAbsolutePath());
        }
    }

    // 删除垃圾文件
    public static void deleteTrashFile(File file) throws IOException {
        if (file.getName().startsWith(".") && !file.delete()) {
            throw new IOException("fail to delete file: " + file.getAbsolutePath());
        }
        if (file.isDirectory()) {
            for (File subFile : file.listFiles()) {
                deleteTrashFile(subFile);
            }
        }

    }
}
