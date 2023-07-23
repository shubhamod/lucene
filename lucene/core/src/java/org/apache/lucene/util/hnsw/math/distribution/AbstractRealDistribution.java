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
package org.apache.lucene.util.hnsw.math.distribution;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.solvers.UnivariateSolverUtils;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public abstract class AbstractRealDistribution
implements RealDistribution, Serializable {
    
    public static final double SOLVER_DEFAULT_ABSOLUTE_ACCURACY = 1e-6;
    
    private static final long serialVersionUID = -38038050983108802L;
     
    @Deprecated
    protected org.apache.lucene.util.hnsw.math.random.RandomDataImpl randomData =
        new org.apache.lucene.util.hnsw.math.random.RandomDataImpl();

    
    protected final RandomGenerator random;

    
    private double solverAbsoluteAccuracy = SOLVER_DEFAULT_ABSOLUTE_ACCURACY;

    
    @Deprecated
    protected AbstractRealDistribution() {
        // Legacy users are only allowed to access the deprecated "randomData".
        // New users are forbidden to use this constructor.
        random = null;
    }
    
    protected AbstractRealDistribution(RandomGenerator rng) {
        random = rng;
    }

    
    @Deprecated
    public double cumulativeProbability(double x0, double x1) throws NumberIsTooLargeException {
        return probability(x0, x1);
    }

    
    public double probability(double x0,
                              double x1) {
        if (x0 > x1) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_ENDPOINT_ABOVE_UPPER_ENDPOINT,
                                                x0, x1, true);
        }
        return cumulativeProbability(x1) - cumulativeProbability(x0);
    }

    
    public double inverseCumulativeProbability(final double p) throws OutOfRangeException {
        /*
         * IMPLEMENTATION NOTES
         * --------------------
         * Where applicable, use is made of the one-sided Chebyshev inequality
         * to bracket the root. This inequality states that
         * P(X - mu >= k * sig) <= 1 / (1 + k^2),
         * mu: mean, sig: standard deviation. Equivalently
         * 1 - P(X < mu + k * sig) <= 1 / (1 + k^2),
         * F(mu + k * sig) >= k^2 / (1 + k^2).
         *
         * For k = sqrt(p / (1 - p)), we find
         * F(mu + k * sig) >= p,
         * and (mu + k * sig) is an upper-bound for the root.
         *
         * Then, introducing Y = -X, mean(Y) = -mu, sd(Y) = sig, and
         * P(Y >= -mu + k * sig) <= 1 / (1 + k^2),
         * P(-X >= -mu + k * sig) <= 1 / (1 + k^2),
         * P(X <= mu - k * sig) <= 1 / (1 + k^2),
         * F(mu - k * sig) <= 1 / (1 + k^2).
         *
         * For k = sqrt((1 - p) / p), we find
         * F(mu - k * sig) <= p,
         * and (mu - k * sig) is a lower-bound for the root.
         *
         * In cases where the Chebyshev inequality does not apply, geometric
         * progressions 1, 2, 4, ... and -1, -2, -4, ... are used to bracket
         * the root.
         */
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }

        double lowerBound = getSupportLowerBound();
        if (p == 0.0) {
            return lowerBound;
        }

        double upperBound = getSupportUpperBound();
        if (p == 1.0) {
            return upperBound;
        }

        final double mu = getNumericalMean();
        final double sig = FastMath.sqrt(getNumericalVariance());
        final boolean chebyshevApplies;
        chebyshevApplies = !(Double.isInfinite(mu) || Double.isNaN(mu) ||
                             Double.isInfinite(sig) || Double.isNaN(sig));

        if (lowerBound == Double.NEGATIVE_INFINITY) {
            if (chebyshevApplies) {
                lowerBound = mu - sig * FastMath.sqrt((1. - p) / p);
            } else {
                lowerBound = -1.0;
                while (cumulativeProbability(lowerBound) >= p) {
                    lowerBound *= 2.0;
                }
            }
        }

        if (upperBound == Double.POSITIVE_INFINITY) {
            if (chebyshevApplies) {
                upperBound = mu + sig * FastMath.sqrt(p / (1. - p));
            } else {
                upperBound = 1.0;
                while (cumulativeProbability(upperBound) < p) {
                    upperBound *= 2.0;
                }
            }
        }

        final UnivariateFunction toSolve = new UnivariateFunction() {
            
            public double value(final double x) {
                return cumulativeProbability(x) - p;
            }
        };

        double x = UnivariateSolverUtils.solve(toSolve,
                                                   lowerBound,
                                                   upperBound,
                                                   getSolverAbsoluteAccuracy());

        if (!isSupportConnected()) {
            /* Test for plateau. */
            final double dx = getSolverAbsoluteAccuracy();
            if (x - dx >= getSupportLowerBound()) {
                double px = cumulativeProbability(x);
                if (cumulativeProbability(x - dx) == px) {
                    upperBound = x;
                    while (upperBound - lowerBound > dx) {
                        final double midPoint = 0.5 * (lowerBound + upperBound);
                        if (cumulativeProbability(midPoint) < px) {
                            lowerBound = midPoint;
                        } else {
                            upperBound = midPoint;
                        }
                    }
                    return upperBound;
                }
            }
        }
        return x;
    }

    
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public void reseedRandomGenerator(long seed) {
        random.setSeed(seed);
        randomData.reSeed(seed);
    }

    
    public double sample() {
        return inverseCumulativeProbability(random.nextDouble());
    }

    
    public double[] sample(int sampleSize) {
        if (sampleSize <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_SAMPLES,
                    sampleSize);
        }
        double[] out = new double[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            out[i] = sample();
        }
        return out;
    }

    
    public double probability(double x) {
        return 0d;
    }

    
    public double logDensity(double x) {
        return FastMath.log(density(x));
    }
}

