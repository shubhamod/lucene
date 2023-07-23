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
package org.apache.lucene.util.hnsw.math.analysis.interpolation;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialSplineFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.exception.NotFiniteNumberException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class LoessInterpolator
    implements UnivariateInterpolator, Serializable {
    
    public static final double DEFAULT_BANDWIDTH = 0.3;
    
    public static final int DEFAULT_ROBUSTNESS_ITERS = 2;
    
    public static final double DEFAULT_ACCURACY = 1e-12;
    
    private static final long serialVersionUID = 5204927143605193821L;
    
    private final double bandwidth;
    
    private final int robustnessIters;
    
    private final double accuracy;

    
    public LoessInterpolator() {
        this.bandwidth = DEFAULT_BANDWIDTH;
        this.robustnessIters = DEFAULT_ROBUSTNESS_ITERS;
        this.accuracy = DEFAULT_ACCURACY;
    }

    
    public LoessInterpolator(double bandwidth, int robustnessIters) {
        this(bandwidth, robustnessIters, DEFAULT_ACCURACY);
    }

    
    public LoessInterpolator(double bandwidth, int robustnessIters, double accuracy)
        throws OutOfRangeException,
               NotPositiveException {
        if (bandwidth < 0 ||
            bandwidth > 1) {
            throw new OutOfRangeException(LocalizedFormats.BANDWIDTH, bandwidth, 0, 1);
        }
        this.bandwidth = bandwidth;
        if (robustnessIters < 0) {
            throw new NotPositiveException(LocalizedFormats.ROBUSTNESS_ITERATIONS, robustnessIters);
        }
        this.robustnessIters = robustnessIters;
        this.accuracy = accuracy;
    }

    
    public final PolynomialSplineFunction interpolate(final double[] xval,
                                                      final double[] yval)
        throws NonMonotonicSequenceException,
               DimensionMismatchException,
               NoDataException,
               NotFiniteNumberException,
               NumberIsTooSmallException {
        return new SplineInterpolator().interpolate(xval, smooth(xval, yval));
    }

    
    public final double[] smooth(final double[] xval, final double[] yval,
                                 final double[] weights)
        throws NonMonotonicSequenceException,
               DimensionMismatchException,
               NoDataException,
               NotFiniteNumberException,
               NumberIsTooSmallException {
        if (xval.length != yval.length) {
            throw new DimensionMismatchException(xval.length, yval.length);
        }

        final int n = xval.length;

        if (n == 0) {
            throw new NoDataException();
        }

        checkAllFiniteReal(xval);
        checkAllFiniteReal(yval);
        checkAllFiniteReal(weights);

        MathArrays.checkOrder(xval);

        if (n == 1) {
            return new double[]{yval[0]};
        }

        if (n == 2) {
            return new double[]{yval[0], yval[1]};
        }

        int bandwidthInPoints = (int) (bandwidth * n);

        if (bandwidthInPoints < 2) {
            throw new NumberIsTooSmallException(LocalizedFormats.BANDWIDTH,
                                                bandwidthInPoints, 2, true);
        }

        final double[] res = new double[n];

        final double[] residuals = new double[n];
        final double[] sortedResiduals = new double[n];

        final double[] robustnessWeights = new double[n];

        // Do an initial fit and 'robustnessIters' robustness iterations.
        // This is equivalent to doing 'robustnessIters+1' robustness iterations
        // starting with all robustness weights set to 1.
        Arrays.fill(robustnessWeights, 1);

        for (int iter = 0; iter <= robustnessIters; ++iter) {
            final int[] bandwidthInterval = {0, bandwidthInPoints - 1};
            // At each x, compute a local weighted linear regression
            for (int i = 0; i < n; ++i) {
                final double x = xval[i];

                // Find out the interval of source points on which
                // a regression is to be made.
                if (i > 0) {
                    updateBandwidthInterval(xval, weights, i, bandwidthInterval);
                }

                final int ileft = bandwidthInterval[0];
                final int iright = bandwidthInterval[1];

                // Compute the point of the bandwidth interval that is
                // farthest from x
                final int edge;
                if (xval[i] - xval[ileft] > xval[iright] - xval[i]) {
                    edge = ileft;
                } else {
                    edge = iright;
                }

                // Compute a least-squares linear fit weighted by
                // the product of robustness weights and the tricube
                // weight function.
                // See http://en.wikipedia.org/wiki/Linear_regression
                // (section "Univariate linear case")
                // and http://en.wikipedia.org/wiki/Weighted_least_squares
                // (section "Weighted least squares")
                double sumWeights = 0;
                double sumX = 0;
                double sumXSquared = 0;
                double sumY = 0;
                double sumXY = 0;
                double denom = FastMath.abs(1.0 / (xval[edge] - x));
                for (int k = ileft; k <= iright; ++k) {
                    final double xk   = xval[k];
                    final double yk   = yval[k];
                    final double dist = (k < i) ? x - xk : xk - x;
                    final double w    = tricube(dist * denom) * robustnessWeights[k] * weights[k];
                    final double xkw  = xk * w;
                    sumWeights += w;
                    sumX += xkw;
                    sumXSquared += xk * xkw;
                    sumY += yk * w;
                    sumXY += yk * xkw;
                }

                final double meanX = sumX / sumWeights;
                final double meanY = sumY / sumWeights;
                final double meanXY = sumXY / sumWeights;
                final double meanXSquared = sumXSquared / sumWeights;

                final double beta;
                if (FastMath.sqrt(FastMath.abs(meanXSquared - meanX * meanX)) < accuracy) {
                    beta = 0;
                } else {
                    beta = (meanXY - meanX * meanY) / (meanXSquared - meanX * meanX);
                }

                final double alpha = meanY - beta * meanX;

                res[i] = beta * x + alpha;
                residuals[i] = FastMath.abs(yval[i] - res[i]);
            }

            // No need to recompute the robustness weights at the last
            // iteration, they won't be needed anymore
            if (iter == robustnessIters) {
                break;
            }

            // Recompute the robustness weights.

            // Find the median residual.
            // An arraycopy and a sort are completely tractable here,
            // because the preceding loop is a lot more expensive
            System.arraycopy(residuals, 0, sortedResiduals, 0, n);
            Arrays.sort(sortedResiduals);
            final double medianResidual = sortedResiduals[n / 2];

            if (FastMath.abs(medianResidual) < accuracy) {
                break;
            }

            for (int i = 0; i < n; ++i) {
                final double arg = residuals[i] / (6 * medianResidual);
                if (arg >= 1) {
                    robustnessWeights[i] = 0;
                } else {
                    final double w = 1 - arg * arg;
                    robustnessWeights[i] = w * w;
                }
            }
        }

        return res;
    }

    
    public final double[] smooth(final double[] xval, final double[] yval)
        throws NonMonotonicSequenceException,
               DimensionMismatchException,
               NoDataException,
               NotFiniteNumberException,
               NumberIsTooSmallException {
        if (xval.length != yval.length) {
            throw new DimensionMismatchException(xval.length, yval.length);
        }

        final double[] unitWeights = new double[xval.length];
        Arrays.fill(unitWeights, 1.0);

        return smooth(xval, yval, unitWeights);
    }

    
    private static void updateBandwidthInterval(final double[] xval, final double[] weights,
                                                final int i,
                                                final int[] bandwidthInterval) {
        final int left = bandwidthInterval[0];
        final int right = bandwidthInterval[1];

        // The right edge should be adjusted if the next point to the right
        // is closer to xval[i] than the leftmost point of the current interval
        int nextRight = nextNonzero(weights, right);
        if (nextRight < xval.length && xval[nextRight] - xval[i] < xval[i] - xval[left]) {
            int nextLeft = nextNonzero(weights, bandwidthInterval[0]);
            bandwidthInterval[0] = nextLeft;
            bandwidthInterval[1] = nextRight;
        }
    }

    
    private static int nextNonzero(final double[] weights, final int i) {
        int j = i + 1;
        while(j < weights.length && weights[j] == 0) {
            ++j;
        }
        return j;
    }

    
    private static double tricube(final double x) {
        final double absX = FastMath.abs(x);
        if (absX >= 1.0) {
            return 0.0;
        }
        final double tmp = 1 - absX * absX * absX;
        return tmp * tmp * tmp;
    }

    
    private static void checkAllFiniteReal(final double[] values) {
        for (int i = 0; i < values.length; i++) {
            MathUtils.checkFinite(values[i]);
        }
    }
}
