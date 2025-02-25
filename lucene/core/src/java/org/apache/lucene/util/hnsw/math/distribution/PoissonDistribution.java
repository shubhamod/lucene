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

import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.special.Gamma;
import org.apache.lucene.util.hnsw.math.util.CombinatoricsUtils;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class PoissonDistribution extends AbstractIntegerDistribution {
    
    public static final int DEFAULT_MAX_ITERATIONS = 10000000;
    
    public static final double DEFAULT_EPSILON = 1e-12;
    
    private static final long serialVersionUID = -3349935121172596109L;
    
    private final NormalDistribution normal;
    
    private final ExponentialDistribution exponential;
    
    private final double mean;

    
    private final int maxIterations;

    
    private final double epsilon;

    
    public PoissonDistribution(double p) throws NotStrictlyPositiveException {
        this(p, DEFAULT_EPSILON, DEFAULT_MAX_ITERATIONS);
    }

    
    public PoissonDistribution(double p, double epsilon, int maxIterations)
    throws NotStrictlyPositiveException {
        this(new Well19937c(), p, epsilon, maxIterations);
    }

    
    public PoissonDistribution(RandomGenerator rng,
                               double p,
                               double epsilon,
                               int maxIterations)
    throws NotStrictlyPositiveException {
        super(rng);

        if (p <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.MEAN, p);
        }
        mean = p;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;

        // Use the same RNG instance as the parent class.
        normal = new NormalDistribution(rng, p, FastMath.sqrt(p),
                                        NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
        exponential = new ExponentialDistribution(rng, 1,
                                                  ExponentialDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public PoissonDistribution(double p, double epsilon)
    throws NotStrictlyPositiveException {
        this(p, epsilon, DEFAULT_MAX_ITERATIONS);
    }

    
    public PoissonDistribution(double p, int maxIterations) {
        this(p, DEFAULT_EPSILON, maxIterations);
    }

    
    public double getMean() {
        return mean;
    }

    
    public double probability(int x) {
        final double logProbability = logProbability(x);
        return logProbability == Double.NEGATIVE_INFINITY ? 0 : FastMath.exp(logProbability);
    }

    
    @Override
    public double logProbability(int x) {
        double ret;
        if (x < 0 || x == Integer.MAX_VALUE) {
            ret = Double.NEGATIVE_INFINITY;
        } else if (x == 0) {
            ret = -mean;
        } else {
            ret = -SaddlePointExpansion.getStirlingError(x) -
                  SaddlePointExpansion.getDeviancePart(x, mean) -
                  0.5 * FastMath.log(MathUtils.TWO_PI) - 0.5 * FastMath.log(x);
        }
        return ret;
    }

    
    public double cumulativeProbability(int x) {
        if (x < 0) {
            return 0;
        }
        if (x == Integer.MAX_VALUE) {
            return 1;
        }
        return Gamma.regularizedGammaQ((double) x + 1, mean, epsilon,
                                       maxIterations);
    }

    
    public double normalApproximateProbability(int x)  {
        // calculate the probability using half-correction
        return normal.cumulativeProbability(x + 0.5);
    }

    
    public double getNumericalMean() {
        return getMean();
    }

    
    public double getNumericalVariance() {
        return getMean();
    }

    
    public int getSupportLowerBound() {
        return 0;
    }

    
    public int getSupportUpperBound() {
        return Integer.MAX_VALUE;
    }

    
    public boolean isSupportConnected() {
        return true;
    }

    
    @Override
    public int sample() {
        return (int) FastMath.min(nextPoisson(mean), Integer.MAX_VALUE);
    }

    
    private long nextPoisson(double meanPoisson) {
        final double pivot = 40.0d;
        if (meanPoisson < pivot) {
            double p = FastMath.exp(-meanPoisson);
            long n = 0;
            double r = 1.0d;
            double rnd = 1.0d;

            while (n < 1000 * meanPoisson) {
                rnd = random.nextDouble();
                r *= rnd;
                if (r >= p) {
                    n++;
                } else {
                    return n;
                }
            }
            return n;
        } else {
            final double lambda = FastMath.floor(meanPoisson);
            final double lambdaFractional = meanPoisson - lambda;
            final double logLambda = FastMath.log(lambda);
            final double logLambdaFactorial = CombinatoricsUtils.factorialLog((int) lambda);
            final long y2 = lambdaFractional < Double.MIN_VALUE ? 0 : nextPoisson(lambdaFractional);
            final double delta = FastMath.sqrt(lambda * FastMath.log(32 * lambda / FastMath.PI + 1));
            final double halfDelta = delta / 2;
            final double twolpd = 2 * lambda + delta;
            final double a1 = FastMath.sqrt(FastMath.PI * twolpd) * FastMath.exp(1 / (8 * lambda));
            final double a2 = (twolpd / delta) * FastMath.exp(-delta * (1 + delta) / twolpd);
            final double aSum = a1 + a2 + 1;
            final double p1 = a1 / aSum;
            final double p2 = a2 / aSum;
            final double c1 = 1 / (8 * lambda);

            double x = 0;
            double y = 0;
            double v = 0;
            int a = 0;
            double t = 0;
            double qr = 0;
            double qa = 0;
            for (;;) {
                final double u = random.nextDouble();
                if (u <= p1) {
                    final double n = random.nextGaussian();
                    x = n * FastMath.sqrt(lambda + halfDelta) - 0.5d;
                    if (x > delta || x < -lambda) {
                        continue;
                    }
                    y = x < 0 ? FastMath.floor(x) : FastMath.ceil(x);
                    final double e = exponential.sample();
                    v = -e - (n * n / 2) + c1;
                } else {
                    if (u > p1 + p2) {
                        y = lambda;
                        break;
                    } else {
                        x = delta + (twolpd / delta) * exponential.sample();
                        y = FastMath.ceil(x);
                        v = -exponential.sample() - delta * (x + 1) / twolpd;
                    }
                }
                a = x < 0 ? 1 : 0;
                t = y * (y + 1) / (2 * lambda);
                if (v < -t && a == 0) {
                    y = lambda + y;
                    break;
                }
                qr = t * ((2 * y + 1) / (6 * lambda) - 1);
                qa = qr - (t * t) / (3 * (lambda + a * (y + 1)));
                if (v < qa) {
                    y = lambda + y;
                    break;
                }
                if (v > qr) {
                    continue;
                }
                if (v < y * logLambda - CombinatoricsUtils.factorialLog((int) (y + lambda)) + logLambdaFactorial) {
                    y = lambda + y;
                    break;
                }
            }
            return y2 + (long) y;
        }
    }
}
