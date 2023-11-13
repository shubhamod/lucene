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

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.ml.distance.DistanceMeasure;
import org.apache.lucene.util.hnsw.math.ml.distance.EuclideanDistance;
import org.apache.lucene.util.hnsw.math.random.JDKRandomGenerator;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class FuzzyKMeansClusterer<T extends Clusterable> extends Clusterer<T> {

    
    private static final double DEFAULT_EPSILON = 1e-3;

    
    private final int k;

    
    private final int maxIterations;

    
    private final double fuzziness;

    
    private final double epsilon;

    
    private final RandomGenerator random;

    
    private double[][] membershipMatrix;

    
    private List<T> points;

    
    private List<CentroidCluster<T>> clusters;

    
    public FuzzyKMeansClusterer(final int k, final double fuzziness) throws NumberIsTooSmallException {
        this(k, fuzziness, -1, new EuclideanDistance());
    }

    
    public FuzzyKMeansClusterer(final int k, final double fuzziness,
                                final int maxIterations, final DistanceMeasure measure)
            throws NumberIsTooSmallException {
        this(k, fuzziness, maxIterations, measure, DEFAULT_EPSILON, new JDKRandomGenerator());
    }

    
    public FuzzyKMeansClusterer(final int k, final double fuzziness,
                                final int maxIterations, final DistanceMeasure measure,
                                final double epsilon, final RandomGenerator random)
            throws NumberIsTooSmallException {

        super(measure);

        if (fuzziness <= 1.0d) {
            throw new NumberIsTooSmallException(fuzziness, 1.0, false);
        }
        this.k = k;
        this.fuzziness = fuzziness;
        this.maxIterations = maxIterations;
        this.epsilon = epsilon;
        this.random = random;

        this.membershipMatrix = null;
        this.points = null;
        this.clusters = null;
    }

    
    public int getK() {
        return k;
    }

    
    public double getFuzziness() {
        return fuzziness;
    }

    
    public int getMaxIterations() {
        return maxIterations;
    }

    
    public double getEpsilon() {
        return epsilon;
    }

    
    public RandomGenerator getRandomGenerator() {
        return random;
    }

    
    public RealMatrix getMembershipMatrix() {
        if (membershipMatrix == null) {
            throw new MathIllegalStateException();
        }
        return MatrixUtils.createRealMatrix(membershipMatrix);
    }

    
    public List<T> getDataPoints() {
        return points;
    }

    
    public List<CentroidCluster<T>> getClusters() {
        return clusters;
    }

    
    public double getObjectiveFunctionValue() {
        if (points == null || clusters == null) {
            throw new MathIllegalStateException();
        }

        int i = 0;
        double objFunction = 0.0;
        for (final T point : points) {
            int j = 0;
            for (final CentroidCluster<T> cluster : clusters) {
                final double dist = distance(point, cluster.getCenter());
                objFunction += (dist * dist) * FastMath.pow(membershipMatrix[i][j], fuzziness);
                j++;
            }
            i++;
        }
        return objFunction;
    }

    
    @Override
    public List<CentroidCluster<T>> cluster(final Collection<T> dataPoints)
            throws MathIllegalArgumentException {

        // sanity checks
        MathUtils.checkNotNull(dataPoints);

        final int size = dataPoints.size();

        // number of clusters has to be smaller or equal the number of data points
        if (size < k) {
            throw new NumberIsTooSmallException(size, k, false);
        }

        // copy the input collection to an unmodifiable list with indexed access
        points = Collections.unmodifiableList(new ArrayList<T>(dataPoints));
        clusters = new ArrayList<CentroidCluster<T>>();
        membershipMatrix = new double[size][k];
        final double[][] oldMatrix = new double[size][k];

        // if no points are provided, return an empty list of clusters
        if (size == 0) {
            return clusters;
        }

        initializeMembershipMatrix();

        // there is at least one point
        final int pointDimension = points.get(0).getPoint().length;
        for (int i = 0; i < k; i++) {
            clusters.add(new CentroidCluster<T>(new DoublePoint(new double[pointDimension])));
        }

        int iteration = 0;
        final int max = (maxIterations < 0) ? Integer.MAX_VALUE : maxIterations;
        double difference = 0.0;

        do {
            saveMembershipMatrix(oldMatrix);
            updateClusterCenters();
            updateMembershipMatrix();
            difference = calculateMaxMembershipChange(oldMatrix);
        } while (difference > epsilon && ++iteration < max);

        return clusters;
    }

    
    private void updateClusterCenters() {
        int j = 0;
        final List<CentroidCluster<T>> newClusters = new ArrayList<CentroidCluster<T>>(k);
        for (final CentroidCluster<T> cluster : clusters) {
            final Clusterable center = cluster.getCenter();
            int i = 0;
            double[] arr = new double[center.getPoint().length];
            double sum = 0.0;
            for (final T point : points) {
                final double u = FastMath.pow(membershipMatrix[i][j], fuzziness);
                final double[] pointArr = point.getPoint();
                for (int idx = 0; idx < arr.length; idx++) {
                    arr[idx] += u * pointArr[idx];
                }
                sum += u;
                i++;
            }
            MathArrays.scaleInPlace(1.0 / sum, arr);
            newClusters.add(new CentroidCluster<T>(new DoublePoint(arr)));
            j++;
        }
        clusters.clear();
        clusters = newClusters;
    }

    
    private void updateMembershipMatrix() {
        for (int i = 0; i < points.size(); i++) {
            final T point = points.get(i);
            double maxMembership = Double.MIN_VALUE;
            int newCluster = -1;
            for (int j = 0; j < clusters.size(); j++) {
                double sum = 0.0;
                final double distA = FastMath.abs(distance(point, clusters.get(j).getCenter()));

                if (distA != 0.0) {
                    for (final CentroidCluster<T> c : clusters) {
                        final double distB = FastMath.abs(distance(point, c.getCenter()));
                        if (distB == 0.0) {
                            sum = Double.POSITIVE_INFINITY;
                            break;
                        }
                        sum += FastMath.pow(distA / distB, 2.0 / (fuzziness - 1.0));
                    }
                }

                double membership;
                if (sum == 0.0) {
                    membership = 1.0;
                } else if (sum == Double.POSITIVE_INFINITY) {
                    membership = 0.0;
                } else {
                    membership = 1.0 / sum;
                }
                membershipMatrix[i][j] = membership;

                if (membershipMatrix[i][j] > maxMembership) {
                    maxMembership = membershipMatrix[i][j];
                    newCluster = j;
                }
            }
            clusters.get(newCluster).addPoint(point);
        }
    }

    
    private void initializeMembershipMatrix() {
        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < k; j++) {
                membershipMatrix[i][j] = random.nextDouble();
            }
            membershipMatrix[i] = MathArrays.normalizeArray(membershipMatrix[i], 1.0);
        }
    }

    
    private double calculateMaxMembershipChange(final double[][] matrix) {
        double maxMembership = 0.0;
        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < clusters.size(); j++) {
                double v = FastMath.abs(membershipMatrix[i][j] - matrix[i][j]);
                maxMembership = FastMath.max(v, maxMembership);
            }
        }
        return maxMembership;
    }

    
    private void saveMembershipMatrix(final double[][] matrix) {
        for (int i = 0; i < points.size(); i++) {
            System.arraycopy(membershipMatrix[i], 0, matrix[i], 0, clusters.size());
        }
    }

}
