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
package org.apache.lucene.util.hnsw.math.ml.clustering;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.ml.distance.DistanceMeasure;
import org.apache.lucene.util.hnsw.math.ml.distance.EuclideanDistance;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class DBSCANClusterer<T extends Clusterable> extends Clusterer<T> {

    
    private final double              eps;

    
    private final int                 minPts;

    
    private enum PointStatus {
        
        NOISE,
        
        PART_OF_CLUSTER
    }

    
    public DBSCANClusterer(final double eps, final int minPts)
        throws NotPositiveException {
        this(eps, minPts, new EuclideanDistance());
    }

    
    public DBSCANClusterer(final double eps, final int minPts, final DistanceMeasure measure)
        throws NotPositiveException {
        super(measure);

        if (eps < 0.0d) {
            throw new NotPositiveException(eps);
        }
        if (minPts < 0) {
            throw new NotPositiveException(minPts);
        }
        this.eps = eps;
        this.minPts = minPts;
    }

    
    public double getEps() {
        return eps;
    }

    
    public int getMinPts() {
        return minPts;
    }

    
    @Override
    public List<Cluster<T>> cluster(final Collection<T> points) throws NullArgumentException {

        // sanity checks
        MathUtils.checkNotNull(points);

        final List<Cluster<T>> clusters = new ArrayList<Cluster<T>>();
        final Map<Clusterable, PointStatus> visited = new HashMap<Clusterable, PointStatus>();

        for (final T point : points) {
            if (visited.get(point) != null) {
                continue;
            }
            final List<T> neighbors = getNeighbors(point, points);
            if (neighbors.size() >= minPts) {
                // DBSCAN does not care about center points
                final Cluster<T> cluster = new Cluster<T>();
                clusters.add(expandCluster(cluster, point, neighbors, points, visited));
            } else {
                visited.put(point, PointStatus.NOISE);
            }
        }

        return clusters;
    }

    
    private Cluster<T> expandCluster(final Cluster<T> cluster,
                                     final T point,
                                     final List<T> neighbors,
                                     final Collection<T> points,
                                     final Map<Clusterable, PointStatus> visited) {
        cluster.addPoint(point);
        visited.put(point, PointStatus.PART_OF_CLUSTER);

        List<T> seeds = new ArrayList<T>(neighbors);
        int index = 0;
        while (index < seeds.size()) {
            final T current = seeds.get(index);
            PointStatus pStatus = visited.get(current);
            // only check non-visited points
            if (pStatus == null) {
                final List<T> currentNeighbors = getNeighbors(current, points);
                if (currentNeighbors.size() >= minPts) {
                    seeds = merge(seeds, currentNeighbors);
                }
            }

            if (pStatus != PointStatus.PART_OF_CLUSTER) {
                visited.put(current, PointStatus.PART_OF_CLUSTER);
                cluster.addPoint(current);
            }

            index++;
        }
        return cluster;
    }

    
    private List<T> getNeighbors(final T point, final Collection<T> points) {
        final List<T> neighbors = new ArrayList<T>();
        for (final T neighbor : points) {
            if (point != neighbor && distance(neighbor, point) <= eps) {
                neighbors.add(neighbor);
            }
        }
        return neighbors;
    }

    
    private List<T> merge(final List<T> one, final List<T> two) {
        final Set<T> oneSet = new HashSet<T>(one);
        for (T item : two) {
            if (!oneSet.contains(item)) {
                one.add(item);
            }
        }
        return one;
    }
}
