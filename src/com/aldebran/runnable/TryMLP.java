package com.aldebran.runnable;

import com.aldebran.algo.iter.CSVDataSetIterator;
import com.aldebran.algo.mlp.MLP;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;

/**
 * MLP demo
 *
 * @author aldebran
 * @since 2021-11-24
 */
public class TryMLP {

    static int in = 2; // 输入向量长度

    static int out = 1; // 输出向量长度

    static int batchSize = 50; // 批大小

    static int seed = 1; // 随机种子

    static int trainEpochs = 500; // 训练周期数

    static String trainFile = "/Users/aldebran/Downloads/000.csv";

    static String testFile = "/Users/aldebran/Downloads/001.csv";

    static String mMapFile = "/Users/aldebran/Downloads/dataset/tmpFile"; // 内存映射文件

    static long mMapFileSize = 40L * 1024 * 1024 * 1024; // 内存映射文件大小

    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("javacpp.platform", "linux-arm64");// windows/linux x86-64不要设置此项

        MLP mlp = generateRandom();
        mlp.train(trainEpochs);

        System.out.println(mlp.regressionEvaluate().stats());
        System.out.println(Arrays.toString(mlp.predict(new double[]{200, -100})));
        System.out.println(Arrays.toString(mlp.predict(new double[]{1, 2})));
        System.out.println(Arrays.toString(mlp.predict(new double[]{300, -300})));
        System.out.println(Arrays.toString(mlp.predict(new double[]{-300, -300})));
        System.out.println(Arrays.toString(mlp.predict(new double[]{-300, +300})));
    }

    // 构建模型，数据集来自文件
    public static MLP fromFile() throws IOException {
        MLP mlp = new MLP(in, out, new File(trainFile), new File(testFile), null,
                StandardCharsets.UTF_8, batchSize, 1);

        mlp.buildNetwork();
        mlp.activateMMapFile(new File(mMapFile), mMapFileSize);
        return mlp;
    }

    // 构建模型，随机生成数据集，模拟 z = 2 * x + y
    public static MLP generateRandom() throws IOException {
        Random random = new Random(seed);
        int trainRecordsCount = 10000;
        int testRecordsCount = 1000;
        CSVDataSetIterator trainIter = new CSVDataSetIterator(randomRecords(random, trainRecordsCount),
                batchSize, in, out);


        CSVDataSetIterator testIter = new CSVDataSetIterator(randomRecords(random, testRecordsCount),
                batchSize, in, out);


        MLP mlp = new MLP(in, out, trainIter, testIter, null, seed);
        mlp.buildNetwork();
        mlp.activateMMapFile(new File(mMapFile), mMapFileSize);
        return mlp;

    }

    private static double[][] randomRecords(Random random, int recordsCount) {
        double[][] data = new double[recordsCount][3];
        for (int i = 0; i < recordsCount; i++) {
            int bound1 = Math.abs(random.nextInt(1000)) + 1;
            int bound2 = Math.abs(random.nextInt(1000)) + 1;

            int a = random.nextInt(bound1);
            int b = random.nextInt(bound2);
            if (random.nextDouble() < 0.5) {
                a = -a;
            }
            if (random.nextDouble() < 0.5) {
                b = -b;
            }
            int c = a + 2 * b;
            data[i][0] = a;
            data[i][1] = b;
            data[i][2] = c;
        }
//        System.out.println(Arrays.deepToString(data));
        return data;
    }
}
