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

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.lucene.util.hnsw.math.analysis.function.Gaussian;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem;
import org.apache.lucene.util.hnsw.math.linear.DiagonalMatrix;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class GaussianCurveFitter extends AbstractCurveFitter {
    
    private static final Gaussian.Parametric FUNCTION = new Gaussian.Parametric() {
            
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
    
    private final double[] initialGuess;
    
    private final int maxIter;

    
    private GaussianCurveFitter(double[] initialGuess,
                                int maxIter) {
        this.initialGuess = initialGuess;
        this.maxIter = maxIter;
    }

    
    public static GaussianCurveFitter create() {
        return new GaussianCurveFitter(null, Integer.MAX_VALUE);
    }

    
    public GaussianCurveFitter withStartPoint(double[] newStart) {
        return new GaussianCurveFitter(newStart.clone(),
                                       maxIter);
    }

    
    public GaussianCurveFitter withMaxIterations(int newMaxIter) {
        return new GaussianCurveFitter(initialGuess,
                                       newMaxIter);
    }

    
    @Override
    protected LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> observations) {

        // Prepare least-squares problem.
        final int len = observations.size();
        final double[] target  = new double[len];
        final double[] weights = new double[len];

        int i = 0;
        for (WeightedObservedPoint obs : observations) {
            target[i]  = obs.getY();
            weights[i] = obs.getWeight();
            ++i;
        }

        final AbstractCurveFitter.TheoreticalValuesFunction model =
                new AbstractCurveFitter.TheoreticalValuesFunction(FUNCTION, observations);

        final double[] startPoint = initialGuess != null ?
            initialGuess :
            // Compute estimation.
            new ParameterGuesser(observations).guess();

        // Return a new least squares problem set up to fit a Gaussian curve to the
        // observed points.
        return new LeastSquaresBuilder().
                maxEvaluations(Integer.MAX_VALUE).
                maxIterations(maxIter).
                start(startPoint).
                target(target).
                weight(new DiagonalMatrix(weights)).
                model(model.getModelFunction(), model.getModelFunctionJacobian()).
                build();

    }

    
    public static class ParameterGuesser {
        
        private final double norm;
        
        private final double mean;
        
        private final double sigma;

        
        public ParameterGuesser(Collection<WeightedObservedPoint> observations) {
            if (observations == null) {
                throw new NullArgumentException(LocalizedFormats.INPUT_ARRAY);
            }
            if (observations.size() < 3) {
                throw new NumberIsTooSmallException(observations.size(), 3, true);
            }

            final List<WeightedObservedPoint> sorted = sortObservations(observations);
            final double[] params = basicGuess(sorted.toArray(new WeightedObservedPoint[0]));

            norm = params[0];
            mean = params[1];
            sigma = params[2];
        }

        
        public double[] guess() {
            return new double[] { norm, mean, sigma };
        }

        
        private List<WeightedObservedPoint> sortObservations(Collection<WeightedObservedPoint> unsorted) {
            final List<WeightedObservedPoint> observations = new ArrayList<WeightedObservedPoint>(unsorted);

            final Comparator<WeightedObservedPoint> cmp = new Comparator<WeightedObservedPoint>() {
                
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

            Collections.sort(observations, cmp);
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
