package com.aldebran.algo;

public class Matrix2D {

    double[][] array;

    int rows;

    int cols;

    Matrix2D(int rows, int cols) {
        array = new double[rows][cols];
        this.rows = rows;
        this.cols = cols;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double get(int i, int j) {
        return array[i][j];
    }

    public void set(int i, int j, double value) {
        array[i][j] = value;
    }
}
