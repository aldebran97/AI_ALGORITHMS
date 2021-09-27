package com.aldebran.algo.k_mean;

import com.aldebran.algo.Vector;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * K均值聚类
 *
 * @author aldebran
 */
public class KMean {

    private List<Vector> vectorList = new ArrayList<>();

    private List<List<Vector>> collections = new ArrayList<>();

    private List<Vector> centroidList = new ArrayList<>();

    public int K;

    public int maxTimes;

    public double threshold;

    public int n;

    /**
     * 构造方法
     *
     * @param K              集群数量
     * @param maxTimes       最大迭代次数，-1表示不限次数
     * @param vectorIterator 向量迭代器
     * @param threshold      临界值，所有质心移动距离小于临界值
     */
    public KMean(int K, int maxTimes, Iterator<Vector> vectorIterator, double threshold) {
        this.K = K;
        this.maxTimes = maxTimes;
        while (vectorIterator.hasNext()) {
            vectorList.add(vectorIterator.next());
        }
        for (int i = 0; i < K; i++) {
            collections.add(new ArrayList<>(K));
            centroidList.add(vectorList.get(i));
        }
        this.threshold = threshold;
        n = vectorList.get(0).getN();
    }

    public KMean(int K, Iterator<Vector> vectorIterator, double threshold) {
        this(K, -1, vectorIterator, threshold);
    }

    /**
     * 运行算法获取结果
     *
     * @return 列表，列表每个元素是个向量集
     */
    public List<List<Vector>> getCollections() {
        for (int t = 0; t < maxTimes || maxTimes == -1; ) {
            for (Vector currentVector : vectorList) {
                double distance = Double.MAX_VALUE;
                int minDistanceVIndex = 0;
                for (int i = 0; i < K; i++) {
                    Vector centroid = centroidList.get(i);
                    double cDistance = Vector.distance(centroid, currentVector);
                    if (cDistance < distance) {
                        distance = cDistance;
                        minDistanceVIndex = i;
                    }
                }
                collections.get(minDistanceVIndex).add(currentVector);
            }
            List<Vector> newCentroidList = new ArrayList<>(K);
            for (int i = 0; i < K; i++) {
                newCentroidList.add(Vector.centroid(collections.get(i).iterator()));
            }
            boolean end = true;
            for (int i = 0; i < K; i++) {
                double distance = Vector.distance(newCentroidList.get(i), centroidList.get(i));
                if (distance > threshold) {
                    end = false;
                    break;
                }
            }
            if (end) {
                System.out.println("总轮次：" + (t + 1));
                break;
            }

            centroidList = newCentroidList;
            for (int i = 0; i < K; i++) {
                collections.get(i).clear();
            }

            if (maxTimes != -1) {
                t++;
            }
        }
        return collections;
    }
}
