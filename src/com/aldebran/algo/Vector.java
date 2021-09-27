package com.aldebran.algo;

import java.util.Arrays;
import java.util.Iterator;

public class Vector extends Matrix2D {

    public Vector(int n) {
        super(1, n);
    }

    public Vector(double... vs) {
        super(1, vs.length);
        for (int i = 0; i < vs.length; i++) {
            set(i, vs[i]);
        }
    }

    public int getN() {
        return getCols();
    }

    public double get(int i) {
        return get(0, i);
    }

    public void set(int i, double value) {
        set(0, i, value);
    }

    // 点之间的距离
    public static double distance(Vector v1, Vector v2) {
        int n;
        if ((n = v1.getN()) != v2.getN()) {
            throw new RuntimeException("the lengths of these 2 vectors are not equal");
        }
        double d = 0;
        for (int i = 0; i < n; i++) {
            d += Math.pow(v1.get(i) - v2.get(i), 2);
        }
        return Math.sqrt(d);
    }

    public Vector add(Vector v) {
        return add(this, v);
    }

    public Vector mul(double k) {
        return mul(this, k);
    }

    public Vector div(double k) {
        return div(this, k);
    }

    // 向量加法
    public static Vector add(Vector v1, Vector v2) {
        int n = v1.getN();
        if (n != v2.getN()) {
            throw new RuntimeException("the lengths of these 2 vectors are not equal");
        }
        Vector v3 = new Vector(n);
        for (int i = 0; i < n; i++) {
            v3.set(i, v1.get(i) + v2.get(i));
        }
        return v3;
    }

    // 数乘
    public static Vector mul(Vector v, double k) {
        int n = v.getN();
        Vector result = new Vector(n);
        for (int i = 0; i < n; i++) {
            result.set(i, k * v.get(i));
        }
        return result;
    }

    public static Vector div(Vector v, double k) {
        return mul(v, 1 / k);
    }


    // 点集合质心
    public static Vector centroid(Iterator<Vector> vectorIterator) {
        Vector result = null;
        Vector v;
        while (vectorIterator.hasNext()) {
            v = vectorIterator.next();
            if (result == null) {
                result = v;
            } else {
                result = add(v, result);
            }
        }
        int n = result.getN();
        for (int i = 0; i < n; i++) {
            result.set(i, result.get(i) / n);
        }
        return result;
    }

    @Override
    public String toString() {
        return Arrays.toString(array[0]);
    }
}
