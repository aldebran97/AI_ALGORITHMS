package com.aldebran.runnable;

import com.aldebran.algo.Vector;
import com.aldebran.algo.k_mean.KMean;

import java.util.Arrays;
import java.util.List;

public class TryKMean {

    // 点集
    static List<Vector> points = Arrays.asList(
            new Vector(0, 0), new Vector(1, 1), new Vector(0, 1),
            new Vector(3, 3), new Vector(3, 4), new Vector(4, 3)
    );

    static int numClusters = 2; // 集群个数

    static double threshold = 0.001; // 阈值

    public static void main(String[] args) {
        KMean kMean = new KMean(numClusters, points.listIterator(), threshold);
        System.out.println(kMean.getCollections());
    }


}
