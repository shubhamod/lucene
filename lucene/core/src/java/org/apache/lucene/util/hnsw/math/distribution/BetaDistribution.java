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

import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.special.Beta;
import org.apache.lucene.util.hnsw.math.special.Gamma;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class BetaDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = -1221965979403477668L;
    
    private final double alpha;
    
    private final double beta;
    
    private double z;
    
    private final double solverAbsoluteAccuracy;

    
    public BetaDistribution(double alpha, double beta) {
        this(alpha, beta, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public BetaDistribution(double alpha, double beta, double inverseCumAccuracy) {
        this(new Well19937c(), alpha, beta, inverseCumAccuracy);
    }

    
    public BetaDistribution(RandomGenerator rng, double alpha, double beta) {
        this(rng, alpha, beta, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public BetaDistribution(RandomGenerator rng,
                            double alpha,
                            double beta,
                            double inverseCumAccuracy) {
        super(rng);

        this.alpha = alpha;
        this.beta = beta;
        z = Double.NaN;
        solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    
    public double getAlpha() {
        return alpha;
    }

    
    public double getBeta() {
        return beta;
    }

    
    private void recomputeZ() {
        if (Double.isNaN(z)) {
            z = Gamma.logGamma(alpha) + Gamma.logGamma(beta) - Gamma.logGamma(alpha + beta);
        }
    }

    
    public double density(double x) {
        final double logDensity = logDensity(x);
        return logDensity == Double.NEGATIVE_INFINITY ? 0 : FastMath.exp(logDensity);
    }

    
    @Override
    public double logDensity(double x) {
        recomputeZ();
        if (x < 0 || x > 1) {
            return Double.NEGATIVE_INFINITY;
        } else if (x == 0) {
            if (alpha < 1) {
                throw new NumberIsTooSmallException(LocalizedFormats.CANNOT_COMPUTE_BETA_DENSITY_AT_0_FOR_SOME_ALPHA, alpha, 1, false);
            }
            return Double.NEGATIVE_INFINITY;
        } else if (x == 1) {
            if (beta < 1) {
                throw new NumberIsTooSmallException(LocalizedFormats.CANNOT_COMPUTE_BETA_DENSITY_AT_1_FOR_SOME_BETA, beta, 1, false);
            }
            return Double.NEGATIVE_INFINITY;
        } else {
            double logX = FastMath.log(x);
            double log1mX = FastMath.log1p(-x);
            return (alpha - 1) * logX + (beta - 1) * log1mX - z;
        }
    }

    
    public double cumulativeProbability(double x)  {
        if (x <= 0) {
            return 0;
        } else if (x >= 1) {
            return 1;
        } else {
            return Beta.regularizedBeta(x, alpha, beta);
        }
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        final double a = getAlpha();
        return a / (a + getBeta());
    }

    
    public double getNumericalVariance() {
        final double a = getAlpha();
        final double b = getBeta();
        final double alphabetasum = a + b;
        return (a * b) / ((alphabetasum * alphabetasum) * (alphabetasum + 1));
    }

    
    public double getSupportLowerBound() {
        return 0;
    }

    
    public double getSupportUpperBound() {
        return 1;
    }

    
    public boolean isSupportLowerBoundInclusive() {
        return false;
    }

    
    public boolean isSupportUpperBoundInclusive() {
        return false;
    }

    
    public boolean isSupportConnected() {
        return true;
    }


    
    @Override
    public double sample() {
        return ChengBetaSampler.sample(random, alpha, beta);
    }

    
    private static final class ChengBetaSampler {

        
        static double sample(RandomGenerator random, final double alpha, final double beta) {
            final double a = FastMath.min(alpha, beta);
            final double b = FastMath.max(alpha, beta);

            if (a > 1) {
                return algorithmBB(random, alpha, a, b);
            } else {
                return algorithmBC(random, alpha, b, a);
            }
        }

        
        private static double algorithmBB(RandomGenerator random,
                                          final double a0,
                                          final double a,
                                          final double b) {
            final double alpha = a + b;
            final double beta = FastMath.sqrt((alpha - 2.) / (2. * a * b - alpha));
            final double gamma = a + 1. / beta;

            double r;
            double w;
            double t;
            do {
                final double u1 = random.nextDouble();
                final double u2 = random.nextDouble();
                final double v = beta * (FastMath.log(u1) - FastMath.log1p(-u1));
                w = a * FastMath.exp(v);
                final double z = u1 * u1 * u2;
                r = gamma * v - 1.3862944;
                final double s = a + r - w;
                if (s + 2.609438 >= 5 * z) {
                    break;
                }

                t = FastMath.log(z);
                if (s >= t) {
                    break;
                }
            } while (r + alpha * (FastMath.log(alpha) - FastMath.log(b + w)) < t);

            w = FastMath.min(w, Double.MAX_VALUE);
            return Precision.equals(a, a0) ? w / (b + w) : b / (b + w);
        }

        
        private static double algorithmBC(RandomGenerator random,
                                          final double a0,
                                          final double a,
                                          final double b) {
            final double alpha = a + b;
            final double beta = 1. / b;
            final double delta = 1. + a - b;
            final double k1 = delta * (0.0138889 + 0.0416667 * b) / (a * beta - 0.777778);
            final double k2 = 0.25 + (0.5 + 0.25 / delta) * b;

            double w;
            for (;;) {
                final double u1 = random.nextDouble();
                final double u2 = random.nextDouble();
                final double y = u1 * u2;
                final double z = u1 * y;
                if (u1 < 0.5) {
                    if (0.25 * u2 + z - y >= k1) {
                        continue;
                    }
                } else {
                    if (z <= 0.25) {
                        final double v = beta * (FastMath.log(u1) - FastMath.log1p(-u1));
                        w = a * FastMath.exp(v);
                        break;
                    }

                    if (z >= k2) {
                        continue;
                    }
                }

                final double v = beta * (FastMath.log(u1) - FastMath.log1p(-u1));
                w = a * FastMath.exp(v);
                if (alpha * (FastMath.log(alpha) - FastMath.log(b + w) + v) - 1.3862944 >= FastMath.log(z)) {
                    break;
                }
            }

            w = FastMath.min(w, Double.MAX_VALUE);
            return Precision.equals(a, a0) ? w / (b + w) : b / (b + w);
        }

    }
}
