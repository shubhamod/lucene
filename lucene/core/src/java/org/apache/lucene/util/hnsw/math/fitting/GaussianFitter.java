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
package org.apache.lucene.util.hnsw.math.fitting;

import java.util.Arrays;
import java.util.Comparator;
import org.apache.lucene.util.hnsw.math.analysis.function.Gaussian;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.MultivariateVectorOptimizer;
import org.apache.lucene.util.hnsw.math.util.FastMath;


@Deprecated
public class GaussianFitter extends CurveFitter<Gaussian.Parametric> {
    
    public GaussianFitter(MultivariateVectorOptimizer optimizer) {
        super(optimizer);
    }

    
    public double[] fit(double[] initialGuess) {
        final Gaussian.Parametric f = new Gaussian.Parametric() {
                
                @Override
                public double value(double x, double ... p) {
                    double v = Double.POSITIVE_INFINITY;
                    try {
                        v = super.value(x, p);
                    } catch (NotStrictlyPositiveException e) { // NOPMD
                        // Do nothing.
                    }
                    return v;
                }

                
                @Override
                public double[] gradient(double x, double ... p) {
                    double[] v = { Double.POSITIVE_INFINITY,
                                   Double.POSITIVE_INFINITY,
                                   Double.POSITIVE_INFINITY };
                    try {
                        v = super.gradient(x, p);
                    } catch (NotStrictlyPositiveException e) { // NOPMD
                        // Do nothing.
                    }
                    return v;
                }
            };

        return fit(f, initialGuess);
    }

    
    public double[] fit() {
        final double[] guess = (new ParameterGuesser(getObservations())).guess();
        return fit(guess);
    }

    
    public static class ParameterGuesser {
        
        private final double norm;
        
        private final double mean;
        
        private final double sigma;

        
        public ParameterGuesser(WeightedObservedPoint[] observations) {
            if (observations == null) {
                throw new NullArgumentException(LocalizedFormats.INPUT_ARRAY);
            }
            if (observations.length < 3) {
                throw new NumberIsTooSmallException(observations.length, 3, true);
            }

            final WeightedObservedPoint[] sorted = sortObservations(observations);
            final double[] params = basicGuess(sorted);

            norm = params[0];
            mean = params[1];
            sigma = params[2];
        }

        
        public double[] guess() {
            return new double[] { norm, mean, sigma };
        }

        
        private WeightedObservedPoint[] sortObservations(WeightedObservedPoint[] unsorted) {
            final WeightedObservedPoint[] observations = unsorted.clone();
            final Comparator<WeightedObservedPoint> cmp
                = new Comparator<WeightedObservedPoint>() {
                
                public int compare(WeightedObservedPoint p1,
                                   WeightedObservedPoint p2) {
                    if (p1 == null && p2 == null) {
                        return 0;
                    }
                    if (p1 == null) {
                        return -1;
                    }
                    if (p2 == null) {
                        return 1;
                    }
                    final int cmpX = Double.compare(p1.getX(), p2.getX());
                    if (cmpX < 0) {
                        return -1;
                    }
                    if (cmpX > 0) {
                        return 1;
                    }
                    final int cmpY = Double.compare(p1.getY(), p2.getY());
                    if (cmpY < 0) {
                        return -1;
                    }
                    if (cmpY > 0) {
                        return 1;
                    }
                    final int cmpW = Double.compare(p1.getWeight(), p2.getWeight());
                    if (cmpW < 0) {
                        return -1;
                    }
                    if (cmpW > 0) {
                        return 1;
                    }
                    return 0;
                }
            };

            Arrays.sort(observations, cmp);
            return observations;
        }

        
        private double[] basicGuess(WeightedObservedPoint[] points) {
            final int maxYIdx = findMaxY(points);
            final double n = points[maxYIdx].getY();
            final double m = points[maxYIdx].getX();

            double fwhmApprox;
            try {
                final double halfY = n + ((m - n) / 2);
                final double fwhmX1 = interpolateXAtY(points, maxYIdx, -1, halfY);
                final double fwhmX2 = interpolateXAtY(points, maxYIdx, 1, halfY);
                fwhmApprox = fwhmX2 - fwhmX1;
            } catch (OutOfRangeException e) {
                // TODO: Exceptions should not be used for flow control.
                fwhmApprox = points[points.length - 1].getX() - points[0].getX();
            }
            final double s = fwhmApprox / (2 * FastMath.sqrt(2 * FastMath.log(2)));

            return new double[] { n, m, s };
        }

        
        private int findMaxY(WeightedObservedPoint[] points) {
            int maxYIdx = 0;
            for (int i = 1; i < points.length; i++) {
                if (points[i].getY() > points[maxYIdx].getY()) {
                    maxYIdx = i;
                }
            }
            return maxYIdx;
        }

        
        private double interpolateXAtY(WeightedObservedPoint[] points,
                                       int startIdx,
                                       int idxStep,
                                       double y)
            throws OutOfRangeException {
            if (idxStep == 0) {
                throw new ZeroException();
            }
            final WeightedObservedPoint[] twoPoints
                = getInterpolationPointsForY(points, startIdx, idxStep, y);
            final WeightedObservedPoint p1 = twoPoints[0];
            final WeightedObservedPoint p2 = twoPoints[1];
            if (p1.getY() == y) {
                return p1.getX();
            }
            if (p2.getY() == y) {
                return p2.getX();
            }
            return p1.getX() + (((y - p1.getY()) * (p2.getX() - p1.getX())) /
                                (p2.getY() - p1.getY()));
        }

        
        private WeightedObservedPoint[] getInterpolationPointsForY(WeightedObservedPoint[] points,
                                                                   int startIdx,
                                                                   int idxStep,
                                                                   double y)
            throws OutOfRangeException {
            if (idxStep == 0) {
                throw new ZeroException();
            }
            for (int i = startIdx;
                 idxStep < 0 ? i + idxStep >= 0 : i + idxStep < points.length;
                 i += idxStep) {
                final WeightedObservedPoint p1 = points[i];
                final WeightedObservedPoint p2 = points[i + idxStep];
                if (isBetween(y, p1.getY(), p2.getY())) {
                    if (idxStep < 0) {
                        return new WeightedObservedPoint[] { p2, p1 };
                    } else {
                        return new WeightedObservedPoint[] { p1, p2 };
                    }
                }
            }

            // Boundaries are replaced by dummy values because the raised
            // exception is caught and the message never displayed.
            // TODO: Exceptions should not be used for flow control.
            throw new OutOfRangeException(y,
                                          Double.NEGATIVE_INFINITY,
                                          Double.POSITIVE_INFINITY);
        }

        
        private boolean isBetween(double value,
                                  double boundary1,
                                  double boundary2) {
            return (value >= boundary1 && value <= boundary2) ||
                (value >= boundary2 && value <= boundary1);
        }
    }
}
