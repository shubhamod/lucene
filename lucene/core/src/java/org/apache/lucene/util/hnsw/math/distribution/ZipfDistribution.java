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
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class ZipfDistribution extends AbstractIntegerDistribution {
    
    private static final long serialVersionUID = -140627372283420404L;
    
    private final int numberOfElements;
    
    private final double exponent;
    
    private double numericalMean = Double.NaN;
    
    private boolean numericalMeanIsCalculated = false;
    
    private double numericalVariance = Double.NaN;
    
    private boolean numericalVarianceIsCalculated = false;
    
    private transient ZipfRejectionInversionSampler sampler;

    
    public ZipfDistribution(final int numberOfElements, final double exponent) {
        this(new Well19937c(), numberOfElements, exponent);
    }

    
    public ZipfDistribution(RandomGenerator rng,
                            int numberOfElements,
                            double exponent)
        throws NotStrictlyPositiveException {
        super(rng);

        if (numberOfElements <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.DIMENSION,
                                                   numberOfElements);
        }
        if (exponent <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.EXPONENT,
                                                   exponent);
        }

        this.numberOfElements = numberOfElements;
        this.exponent = exponent;
    }

    
    public int getNumberOfElements() {
        return numberOfElements;
    }

    
    public double getExponent() {
        return exponent;
    }

    
    public double probability(final int x) {
        if (x <= 0 || x > numberOfElements) {
            return 0.0;
        }

        return (1.0 / FastMath.pow(x, exponent)) / generalizedHarmonic(numberOfElements, exponent);
    }

    
    @Override
    public double logProbability(int x) {
        if (x <= 0 || x > numberOfElements) {
            return Double.NEGATIVE_INFINITY;
        }

        return -FastMath.log(x) * exponent - FastMath.log(generalizedHarmonic(numberOfElements, exponent));
    }

    
    public double cumulativeProbability(final int x) {
        if (x <= 0) {
            return 0.0;
        } else if (x >= numberOfElements) {
            return 1.0;
        }

        return generalizedHarmonic(x, exponent) / generalizedHarmonic(numberOfElements, exponent);
    }

    
    public double getNumericalMean() {
        if (!numericalMeanIsCalculated) {
            numericalMean = calculateNumericalMean();
            numericalMeanIsCalculated = true;
        }
        return numericalMean;
    }

    
    protected double calculateNumericalMean() {
        final int N = getNumberOfElements();
        final double s = getExponent();

        final double Hs1 = generalizedHarmonic(N, s - 1);
        final double Hs = generalizedHarmonic(N, s);

        return Hs1 / Hs;
    }

    
    public double getNumericalVariance() {
        if (!numericalVarianceIsCalculated) {
            numericalVariance = calculateNumericalVariance();
            numericalVarianceIsCalculated = true;
        }
        return numericalVariance;
    }

    
    protected double calculateNumericalVariance() {
        final int N = getNumberOfElements();
        final double s = getExponent();

        final double Hs2 = generalizedHarmonic(N, s - 2);
        final double Hs1 = generalizedHarmonic(N, s - 1);
        final double Hs = generalizedHarmonic(N, s);

        return (Hs2 / Hs) - ((Hs1 * Hs1) / (Hs * Hs));
    }

    
    private double generalizedHarmonic(final int n, final double m) {
        double value = 0;
        for (int k = n; k > 0; --k) {
            value += 1.0 / FastMath.pow(k, m);
        }
        return value;
    }

    
    public int getSupportLowerBound() {
        return 1;
    }

    
    public int getSupportUpperBound() {
        return getNumberOfElements();
    }

    
    public boolean isSupportConnected() {
        return true;
    }

    
    @Override
    public int sample() {
        if (sampler == null) {
            sampler = new ZipfRejectionInversionSampler(numberOfElements, exponent);
        }
        return sampler.sample(random);
    }

    
    static final class ZipfRejectionInversionSampler {

        
        private final double exponent;
        
        private final int numberOfElements;
        
        private final double hIntegralX1;
        
        private final double hIntegralNumberOfElements;
        
        private final double s;

        
        ZipfRejectionInversionSampler(final int numberOfElements, final double exponent) {
            this.exponent = exponent;
            this.numberOfElements = numberOfElements;
            this.hIntegralX1 = hIntegral(1.5) - 1d;
            this.hIntegralNumberOfElements = hIntegral(numberOfElements + 0.5);
            this.s = 2d - hIntegralInverse(hIntegral(2.5) - h(2));
        }

        
        int sample(final RandomGenerator random) {
            while(true) {

                final double u = hIntegralNumberOfElements + random.nextDouble() * (hIntegralX1 - hIntegralNumberOfElements);
                // u is uniformly distributed in (hIntegralX1, hIntegralNumberOfElements]

                double x = hIntegralInverse(u);

                int k = (int)(x + 0.5);

                // Limit k to the range [1, numberOfElements]
                // (k could be outside due to numerical inaccuracies)
                if (k < 1) {
                    k = 1;
                }
                else if (k > numberOfElements) {
                    k = numberOfElements;
                }

                // Here, the distribution of k is given by:
                //
                //   P(k = 1) = C * (hIntegral(1.5) - hIntegralX1) = C
                //   P(k = m) = C * (hIntegral(m + 1/2) - hIntegral(m - 1/2)) for m >= 2
                //
                //   where C := 1 / (hIntegralNumberOfElements - hIntegralX1)

                if (k - x <= s || u >= hIntegral(k + 0.5) - h(k)) {

                    // Case k = 1:
                    //
                    //   The right inequality is always true, because replacing k by 1 gives
                    //   u >= hIntegral(1.5) - h(1) = hIntegralX1 and u is taken from
                    //   (hIntegralX1, hIntegralNumberOfElements].
                    //
                    //   Therefore, the acceptance rate for k = 1 is P(accepted | k = 1) = 1
                    //   and the probability that 1 is returned as random value is
                    //   P(k = 1 and accepted) = P(accepted | k = 1) * P(k = 1) = C = C / 1^exponent
                    //
                    // Case k >= 2:
                    //
                    //   The left inequality (k - x <= s) is just a short cut
                    //   to avoid the more expensive evaluation of the right inequality
                    //   (u >= hIntegral(k + 0.5) - h(k)) in many cases.
                    //
                    //   If the left inequality is true, the right inequality is also true:
                    //     Theorem 2 in the paper is valid for all positive exponents, because
                    //     the requirements h'(x) = -exponent/x^(exponent + 1) < 0 and
                    //     (-1/hInverse'(x))'' = (1+1/exponent) * x^(1/exponent-1) >= 0
                    //     are both fulfilled.
                    //     Therefore, f(x) := x - hIntegralInverse(hIntegral(x + 0.5) - h(x))
                    //     is a non-decreasing function. If k - x <= s holds,
                    //     k - x <= s + f(k) - f(2) is obviously also true which is equivalent to
                    //     -x <= -hIntegralInverse(hIntegral(k + 0.5) - h(k)),
                    //     -hIntegralInverse(u) <= -hIntegralInverse(hIntegral(k + 0.5) - h(k)),
                    //     and finally u >= hIntegral(k + 0.5) - h(k).
                    //
                    //   Hence, the right inequality determines the acceptance rate:
                    //   P(accepted | k = m) = h(m) / (hIntegrated(m+1/2) - hIntegrated(m-1/2))
                    //   The probability that m is returned is given by
                    //   P(k = m and accepted) = P(accepted | k = m) * P(k = m) = C * h(m) = C / m^exponent.
                    //
                    // In both cases the probabilities are proportional to the probability mass function
                    // of the Zipf distribution.

                    return k;
                }
            }
        }

        
        private double hIntegral(final double x) {
            final double logX = FastMath.log(x);
            return helper2((1d-exponent)*logX)*logX;
        }

        
        private double h(final double x) {
            return FastMath.exp(-exponent * FastMath.log(x));
        }

        
        private double hIntegralInverse(final double x) {
            double t = x*(1d-exponent);
            if (t < -1d) {
                // Limit value to the range [-1, +inf).
                // t could be smaller than -1 in some rare cases due to numerical errors.
                t = -1;
            }
            return FastMath.exp(helper1(t)*x);
        }

        
        static double helper1(final double x) {
            if (FastMath.abs(x)>1e-8) {
                return FastMath.log1p(x)/x;
            }
            else {
                return 1.-x*((1./2.)-x*((1./3.)-x*(1./4.)));
            }
        }

        
        static double helper2(final double x) {
            if (FastMath.abs(x)>1e-8) {
                return FastMath.expm1(x)/x;
            }
            else {
                return 1.+x*(1./2.)*(1.+x*(1./3.)*(1.+x*(1./4.)));
            }
        }
    }
}
