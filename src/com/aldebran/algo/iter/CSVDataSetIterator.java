package com.aldebran.algo.iter;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * csv数据集迭代器
 *
 * @author aldebran
 * @since 2021-10-23
 */
public class CSVDataSetIterator implements DataSetIterator {

    private double[][] data;

    private int cols = 0;

    private int rows = 0;

    private int batchSize;

    private int inputNum;

    private int outputNum;

    private int index = 0;

    private boolean finish = false;

    private DataSetPreProcessor dataSetPreProcessor;

    private void init(List<String> lines, int batchSize, int inputNum, int outputNum) {
        this.batchSize = batchSize;
        this.inputNum = inputNum;
        this.outputNum = outputNum;
        rows = lines.size();
        if (rows < 1) {
            throw new RuntimeException("no data!");
        }
        cols = lines.get(0).split(",").length;
        if (cols != inputNum + outputNum) {
            throw new RuntimeException(String.format("illegal size! input: %s, output:%s, cols: %s",
                    inputNum, outputNum, cols));
        }
        data = new double[rows][cols];
        for (int i = 0; i < lines.size(); i++) {
            String[] sp = lines.get(i).split(",");
            for (int j = 0; j < sp.length; j++) {
                data[i][j] = Double.valueOf(sp[j]);
            }
        }

//        System.out.println(rows);
//        System.out.println(cols);
//        System.out.println(Arrays.deepToString(data));

    }

    public CSVDataSetIterator(InputStream inputStream, Charset charset, int batchSize,
                              int inputNum, int outputNum) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, charset));
        try {
            String line = null;
            List<String> lines = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
            init(lines, batchSize, inputNum, outputNum);
        } finally {
            reader.close();
        }
    }


    public CSVDataSetIterator(File file, Charset charset, int batchSize,
                              int inputNum, int outputNum) throws IOException {

        this(new FileInputStream(file), charset, batchSize, inputNum, outputNum);
    }

    public CSVDataSetIterator(double[][] data, int batchSize, int inputNum, int outputNum) {
        this.data = data;
        this.batchSize = batchSize;
        this.inputNum = inputNum;
        this.outputNum = outputNum;
        rows = data.length;
        cols = data[0].length;
    }


    @Override
    public DataSet next(int bs) {
        INDArray input = Nd4j.zeros(new int[]{batchSize, inputNum});

        INDArray label = Nd4j.zeros(new int[]{batchSize, outputNum});

        for (int bI = 0; bI < bs; bI++) {
            int i = (bI + index) % rows;
            if (i >= rows) {
                i = i % rows;
            }
            for (int j = 0; j < inputNum; j++) {
                input.putScalar(new int[]{bI, j}, data[i][j]);
            }

            for (int k = 0, j = inputNum; j < outputNum + inputNum; j++, k++) {
                label.putScalar(new int[]{bI, k}, data[i][j]);
            }
        }
        index += bs;
        if (index >= rows) {
            finish = true;
            index = 0;
        }
//        System.out.println("DataSet");
//        System.out.println("input: " + input);
//        System.out.println("label: " + label);
        DataSet dataset = new DataSet(input, label);
        if (dataSetPreProcessor != null) {
            dataSetPreProcessor.preProcess(dataset);
        }
//        System.out.println(dataset);
        return dataset;
    }

    @Override
    public int inputColumns() {
        return inputNum;
    }

    @Override
    public int totalOutcomes() {
        return outputNum;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        finish = false;
        index = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        this.dataSetPreProcessor = dataSetPreProcessor;
//        throw new UnsupportedOperationException("unsupport");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return dataSetPreProcessor;
//        throw new UnsupportedOperationException("unsupport");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("unsupport");
    }

    @Override
    public boolean hasNext() {
        return !finish;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
