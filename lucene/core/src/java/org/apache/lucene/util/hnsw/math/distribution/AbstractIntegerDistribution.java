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

import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public abstract class AbstractIntegerDistribution implements IntegerDistribution, Serializable {

    
    private static final long serialVersionUID = -1146319659338487221L;

    
    @Deprecated
    protected final org.apache.lucene.util.hnsw.math.random.RandomDataImpl randomData =
        new org.apache.lucene.util.hnsw.math.random.RandomDataImpl();

    
    protected final RandomGenerator random;

    
    @Deprecated
    protected AbstractIntegerDistribution() {
        // Legacy users are only allowed to access the deprecated "randomData".
        // New users are forbidden to use this constructor.
        random = null;
    }

    
    protected AbstractIntegerDistribution(RandomGenerator rng) {
        random = rng;
    }

    
    public double cumulativeProbability(int x0, int x1) throws NumberIsTooLargeException {
        if (x1 < x0) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_ENDPOINT_ABOVE_UPPER_ENDPOINT,
                    x0, x1, true);
        }
        return cumulativeProbability(x1) - cumulativeProbability(x0);
    }

    
    public int inverseCumulativeProbability(final double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }

        int lower = getSupportLowerBound();
        if (p == 0.0) {
            return lower;
        }
        if (lower == Integer.MIN_VALUE) {
            if (checkedCumulativeProbability(lower) >= p) {
                return lower;
            }
        } else {
            lower -= 1; // this ensures cumulativeProbability(lower) < p, which
                        // is important for the solving step
        }

        int upper = getSupportUpperBound();
        if (p == 1.0) {
            return upper;
        }

        // use the one-sided Chebyshev inequality to narrow the bracket
        // cf. AbstractRealDistribution.inverseCumulativeProbability(double)
        final double mu = getNumericalMean();
        final double sigma = FastMath.sqrt(getNumericalVariance());
        final boolean chebyshevApplies = !(Double.isInfinite(mu) || Double.isNaN(mu) ||
                Double.isInfinite(sigma) || Double.isNaN(sigma) || sigma == 0.0);
        if (chebyshevApplies) {
            double k = FastMath.sqrt((1.0 - p) / p);
            double tmp = mu - k * sigma;
            if (tmp > lower) {
                lower = ((int) FastMath.ceil(tmp)) - 1;
            }
            k = 1.0 / k;
            tmp = mu + k * sigma;
            if (tmp < upper) {
                upper = ((int) FastMath.ceil(tmp)) - 1;
            }
        }

        return solveInverseCumulativeProbability(p, lower, upper);
    }

    
    protected int solveInverseCumulativeProbability(final double p, int lower, int upper) {
        while (lower + 1 < upper) {
            int xm = (lower + upper) / 2;
            if (xm < lower || xm > upper) {
                /*
                 * Overflow.
                 * There will never be an overflow in both calculation methods
                 * for xm at the same time
                 */
                xm = lower + (upper - lower) / 2;
            }

            double pm = checkedCumulativeProbability(xm);
            if (pm >= p) {
                upper = xm;
            } else {
                lower = xm;
            }
        }
        return upper;
    }

    
    public void reseedRandomGenerator(long seed) {
        random.setSeed(seed);
        randomData.reSeed(seed);
    }

    
    public int sample() {
        return inverseCumulativeProbability(random.nextDouble());
    }

    
    public int[] sample(int sampleSize) {
        if (sampleSize <= 0) {
            throw new NotStrictlyPositiveException(
                    LocalizedFormats.NUMBER_OF_SAMPLES, sampleSize);
        }
        int[] out = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            out[i] = sample();
        }
        return out;
    }

    
    private double checkedCumulativeProbability(int argument)
        throws MathInternalError {
        double result = Double.NaN;
        result = cumulativeProbability(argument);
        if (Double.isNaN(result)) {
            throw new MathInternalError(LocalizedFormats
                    .DISCRETE_CUMULATIVE_PROBABILITY_RETURNED_NAN, argument);
        }
        return result;
    }

    
    public double logProbability(int x) {
        return FastMath.log(probability(x));
    }
}
