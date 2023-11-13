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
import java.util.Collections;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.ml.distance.DistanceMeasure;
import org.apache.lucene.util.hnsw.math.ml.distance.EuclideanDistance;
import org.apache.lucene.util.hnsw.math.random.JDKRandomGenerator;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Variance;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class KMeansPlusPlusClusterer<T extends Clusterable> extends Clusterer<T> {

    
    public enum EmptyClusterStrategy {

        
        LARGEST_VARIANCE,

        
        LARGEST_POINTS_NUMBER,

        
        FARTHEST_POINT,

        
        ERROR

    }

    
    private final int k;

    
    private final int maxIterations;

    
    private final RandomGenerator random;

    
    private final EmptyClusterStrategy emptyStrategy;

    
    public KMeansPlusPlusClusterer(final int k) {
        this(k, -1);
    }

    
    public KMeansPlusPlusClusterer(final int k, final int maxIterations) {
        this(k, maxIterations, new EuclideanDistance());
    }

    
    public KMeansPlusPlusClusterer(final int k, final int maxIterations, final DistanceMeasure measure) {
        this(k, maxIterations, measure, new JDKRandomGenerator());
    }

    
    public KMeansPlusPlusClusterer(final int k, final int maxIterations,
                                   final DistanceMeasure measure,
                                   final RandomGenerator random) {
        this(k, maxIterations, measure, random, EmptyClusterStrategy.LARGEST_VARIANCE);
    }

    
    public KMeansPlusPlusClusterer(final int k, final int maxIterations,
                                   final DistanceMeasure measure,
                                   final RandomGenerator random,
                                   final EmptyClusterStrategy emptyStrategy) {
        super(measure);
        this.k             = k;
        this.maxIterations = maxIterations;
        this.random        = random;
        this.emptyStrategy = emptyStrategy;
    }

    
    public int getK() {
        return k;
    }

    
    public int getMaxIterations() {
        return maxIterations;
    }

    
    public RandomGenerator getRandomGenerator() {
        return random;
    }

    
    public EmptyClusterStrategy getEmptyClusterStrategy() {
        return emptyStrategy;
    }

    
    @Override
    public List<CentroidCluster<T>> cluster(final Collection<T> points)
        throws MathIllegalArgumentException, ConvergenceException {

        // sanity checks
        MathUtils.checkNotNull(points);

        // number of clusters has to be smaller or equal the number of data points
        if (points.size() < k) {
            throw new NumberIsTooSmallException(points.size(), k, false);
        }

        // create the initial clusters
        List<CentroidCluster<T>> clusters = chooseInitialCenters(points);

        // create an array containing the latest assignment of a point to a cluster
        // no need to initialize the array, as it will be filled with the first assignment
        int[] assignments = new int[points.size()];
        assignPointsToClusters(clusters, points, assignments);

        // iterate through updating the centers until we're done
        final int max = (maxIterations < 0) ? Integer.MAX_VALUE : maxIterations;
        for (int count = 0; count < max; count++) {
            boolean emptyCluster = false;
            List<CentroidCluster<T>> newClusters = new ArrayList<CentroidCluster<T>>();
            for (final CentroidCluster<T> cluster : clusters) {
                final Clusterable newCenter;
                if (cluster.getPoints().isEmpty()) {
                    switch (emptyStrategy) {
                        case LARGEST_VARIANCE :
                            newCenter = getPointFromLargestVarianceCluster(clusters);
                            break;
                        case LARGEST_POINTS_NUMBER :
                            newCenter = getPointFromLargestNumberCluster(clusters);
                            break;
                        case FARTHEST_POINT :
                            newCenter = getFarthestPoint(clusters);
                            break;
                        default :
                            throw new ConvergenceException(LocalizedFormats.EMPTY_CLUSTER_IN_K_MEANS);
                    }
                    emptyCluster = true;
                } else {
                    newCenter = centroidOf(cluster.getPoints(), cluster.getCenter().getPoint().length);
                }
                newClusters.add(new CentroidCluster<T>(newCenter));
            }
            int changes = assignPointsToClusters(newClusters, points, assignments);
            clusters = newClusters;

            // if there were no more changes in the point-to-cluster assignment
            // and there are no empty clusters left, return the current clusters
            if (changes == 0 && !emptyCluster) {
                return clusters;
            }
        }
        return clusters;
    }

    
    private int assignPointsToClusters(final List<CentroidCluster<T>> clusters,
                                       final Collection<T> points,
                                       final int[] assignments) {
        int assignedDifferently = 0;
        int pointIndex = 0;
        for (final T p : points) {
            int clusterIndex = getNearestCluster(clusters, p);
            if (clusterIndex != assignments[pointIndex]) {
                assignedDifferently++;
            }

            CentroidCluster<T> cluster = clusters.get(clusterIndex);
            cluster.addPoint(p);
            assignments[pointIndex++] = clusterIndex;
        }

        return assignedDifferently;
    }

    
    private List<CentroidCluster<T>> chooseInitialCenters(final Collection<T> points) {

        // Convert to list for indexed access. Make it unmodifiable, since removal of items
        // would screw up the logic of this method.
        final List<T> pointList = Collections.unmodifiableList(new ArrayList<T> (points));

        // The number of points in the list.
        final int numPoints = pointList.size();

        // Set the corresponding element in this array to indicate when
        // elements of pointList are no longer available.
        final boolean[] taken = new boolean[numPoints];

        // The resulting list of initial centers.
        final List<CentroidCluster<T>> resultSet = new ArrayList<CentroidCluster<T>>();

        // Choose one center uniformly at random from among the data points.
        final int firstPointIndex = random.nextInt(numPoints);

        final T firstPoint = pointList.get(firstPointIndex);

        resultSet.add(new CentroidCluster<T>(firstPoint));

        // Must mark it as taken
        taken[firstPointIndex] = true;

        // To keep track of the minimum distance squared of elements of
        // pointList to elements of resultSet.
        final double[] minDistSquared = new double[numPoints];

        // Initialize the elements.  Since the only point in resultSet is firstPoint,
        // this is very easy.
        for (int i = 0; i < numPoints; i++) {
            if (i != firstPointIndex) { // That point isn't considered
                double d = distance(firstPoint, pointList.get(i));
                minDistSquared[i] = d*d;
            }
        }

        while (resultSet.size() < k) {

            // Sum up the squared distances for the points in pointList not
            // already taken.
            double distSqSum = 0.0;

            for (int i = 0; i < numPoints; i++) {
                if (!taken[i]) {
                    distSqSum += minDistSquared[i];
                }
            }

            // Add one new data point as a center. Each point x is chosen with
            // probability proportional to D(x)2
            final double r = random.nextDouble() * distSqSum;

            // The index of the next point to be added to the resultSet.
            int nextPointIndex = -1;

            // Sum through the squared min distances again, stopping when
            // sum >= r.
            double sum = 0.0;
            for (int i = 0; i < numPoints; i++) {
                if (!taken[i]) {
                    sum += minDistSquared[i];
                    if (sum >= r) {
                        nextPointIndex = i;
                        break;
                    }
                }
            }

            // If it's not set to >= 0, the point wasn't found in the previous
            // for loop, probably because distances are extremely small.  Just pick
            // the last available point.
            if (nextPointIndex == -1) {
                for (int i = numPoints - 1; i >= 0; i--) {
                    if (!taken[i]) {
                        nextPointIndex = i;
                        break;
                    }
                }
            }

            // We found one.
            if (nextPointIndex >= 0) {

                final T p = pointList.get(nextPointIndex);

                resultSet.add(new CentroidCluster<T> (p));

                // Mark it as taken.
                taken[nextPointIndex] = true;

                if (resultSet.size() < k) {
                    // Now update elements of minDistSquared.  We only have to compute
                    // the distance to the new center to do this.
                    for (int j = 0; j < numPoints; j++) {
                        // Only have to worry about the points still not taken.
                        if (!taken[j]) {
                            double d = distance(p, pointList.get(j));
                            double d2 = d * d;
                            if (d2 < minDistSquared[j]) {
                                minDistSquared[j] = d2;
                            }
                        }
                    }
                }

            } else {
                // None found --
                // Break from the while loop to prevent
                // an infinite loop.
                break;
            }
        }

        return resultSet;
    }

    
    private T getPointFromLargestVarianceCluster(final Collection<CentroidCluster<T>> clusters)
            throws ConvergenceException {

        double maxVariance = Double.NEGATIVE_INFINITY;
        Cluster<T> selected = null;
        for (final CentroidCluster<T> cluster : clusters) {
            if (!cluster.getPoints().isEmpty()) {

                // compute the distance variance of the current cluster
                final Clusterable center = cluster.getCenter();
                final Variance stat = new Variance();
                for (final T point : cluster.getPoints()) {
                    stat.increment(distance(point, center));
                }
                final double variance = stat.getResult();

                // select the cluster with the largest variance
                if (variance > maxVariance) {
                    maxVariance = variance;
                    selected = cluster;
                }

            }
        }

        // did we find at least one non-empty cluster ?
        if (selected == null) {
            throw new ConvergenceException(LocalizedFormats.EMPTY_CLUSTER_IN_K_MEANS);
        }

        // extract a random point from the cluster
        final List<T> selectedPoints = selected.getPoints();
        return selectedPoints.remove(random.nextInt(selectedPoints.size()));

    }

    
    private T getPointFromLargestNumberCluster(final Collection<? extends Cluster<T>> clusters)
            throws ConvergenceException {

        int maxNumber = 0;
        Cluster<T> selected = null;
        for (final Cluster<T> cluster : clusters) {

            // get the number of points of the current cluster
            final int number = cluster.getPoints().size();

            // select the cluster with the largest number of points
            if (number > maxNumber) {
                maxNumber = number;
                selected = cluster;
            }

        }

        // did we find at least one non-empty cluster ?
        if (selected == null) {
            throw new ConvergenceException(LocalizedFormats.EMPTY_CLUSTER_IN_K_MEANS);
        }

        // extract a random point from the cluster
        final List<T> selectedPoints = selected.getPoints();
        return selectedPoints.remove(random.nextInt(selectedPoints.size()));

    }

    
    private T getFarthestPoint(final Collection<CentroidCluster<T>> clusters) throws ConvergenceException {

        double maxDistance = Double.NEGATIVE_INFINITY;
        Cluster<T> selectedCluster = null;
        int selectedPoint = -1;
        for (final CentroidCluster<T> cluster : clusters) {

            // get the farthest point
            final Clusterable center = cluster.getCenter();
            final List<T> points = cluster.getPoints();
            for (int i = 0; i < points.size(); ++i) {
                final double distance = distance(points.get(i), center);
                if (distance > maxDistance) {
                    maxDistance     = distance;
                    selectedCluster = cluster;
                    selectedPoint   = i;
                }
            }

        }

        // did we find at least one non-empty cluster ?
        if (selectedCluster == null) {
            throw new ConvergenceException(LocalizedFormats.EMPTY_CLUSTER_IN_K_MEANS);
        }

        return selectedCluster.getPoints().remove(selectedPoint);

    }

    
    private int getNearestCluster(final Collection<CentroidCluster<T>> clusters, final T point) {
        double minDistance = Double.MAX_VALUE;
        int clusterIndex = 0;
        int minCluster = 0;
        for (final CentroidCluster<T> c : clusters) {
            final double distance = distance(point, c.getCenter());
            if (distance < minDistance) {
                minDistance = distance;
                minCluster = clusterIndex;
            }
            clusterIndex++;
        }
        return minCluster;
    }

    
    private Clusterable centroidOf(final Collection<T> points, final int dimension) {
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
