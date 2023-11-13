/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.util.hnsw.math.ml.clustering.evaluation;

import java.util.List;

import org.apache.lucene.util.hnsw.math.ml.clustering.CentroidCluster;
import org.apache.lucene.util.hnsw.math.ml.clustering.Cluster;
import org.apache.lucene.util.hnsw.math.ml.clustering.Clusterable;
import org.apache.lucene.util.hnsw.math.ml.clustering.DoublePoint;
import org.apache.lucene.util.hnsw.math.ml.distance.DistanceMeasure;
import org.apache.lucene.util.hnsw.math.ml.distance.EuclideanDistance;


public abstract class ClusterEvaluator<T extends Clusterable> {

    
    private final DistanceMeasure measure;

    
    public ClusterEvaluator() {
        this(new EuclideanDistance());
    }

    
    public ClusterEvaluator(final DistanceMeasure measure) {
        this.measure = measure;
    }

    
    public abstract double score(List<? extends Cluster<T>> clusters);

    
    public boolean isBetterScore(double score1, double score2) {
        return score1 < score2;
    }

    
    protected double distance(final Clusterable p1, final Clusterable p2) {
        return measure.compute(p1.getPoint(), p2.getPoint());
    }

    
    protected Clusterable centroidOf(final Cluster<T> cluster) {
        final List<T> points = cluster.getPoints();
        if (points.isEmpty()) {
            return null;
        }

        // in case the cluster is of type CentroidCluster, no need to compute the centroid
        if (cluster instanceof CentroidCluster) {
            return ((CentroidCluster<T>) cluster).getCenter();
        }

        final int dimension = points.get(0).getPoint().length;
        final double[] centroid = new double[dimension];
        for (final T p : points) {
            final double[] point = p.getPoint();
            for (int i = 0; i < centroid.length; i++) {
                centroid[i] += point[i];
            }
        }
        for (int i = 0; i < centroid.length; i++) {
            centroid[i] /= points.size();
        }
        return new DoublePoint(centroid);
    }

}
