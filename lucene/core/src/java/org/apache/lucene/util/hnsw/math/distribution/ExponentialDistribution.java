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
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.CombinatoricsUtils;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.ResizableDoubleArray;


public class ExponentialDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = 2401296428283614780L;
    
    private static final double[] EXPONENTIAL_SA_QI;
    
    private final double mean;
    
    private final double logMean;
    
    private final double solverAbsoluteAccuracy;

    
    static {
        
        final double LN2 = FastMath.log(2);
        double qi = 0;
        int i = 1;

        
        final ResizableDoubleArray ra = new ResizableDoubleArray(20);

        while (qi < 1) {
            qi += FastMath.pow(LN2, i) / CombinatoricsUtils.factorial(i);
            ra.addElement(qi);
            ++i;
        }

        EXPONENTIAL_SA_QI = ra.getElements();
    }

    
    public ExponentialDistribution(double mean) {
        this(mean, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public ExponentialDistribution(double mean, double inverseCumAccuracy) {
        this(new Well19937c(), mean, inverseCumAccuracy);
    }

    
    public ExponentialDistribution(RandomGenerator rng, double mean)
        throws NotStrictlyPositiveException {
        this(rng, mean, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public ExponentialDistribution(RandomGenerator rng,
                                   double mean,
                                   double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        super(rng);

        if (mean <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.MEAN, mean);
        }
        this.mean = mean;
        logMean = FastMath.log(mean);
        solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    
    public double getMean() {
        return mean;
    }

    
    public double density(double x) {
        final double logDensity = logDensity(x);
        return logDensity == Double.NEGATIVE_INFINITY ? 0 : FastMath.exp(logDensity);
    }

    
    @Override
    public double logDensity(double x) {
        if (x < 0) {
            return Double.NEGATIVE_INFINITY;
        }
        return -x / mean - logMean;
    }

    
    public double cumulativeProbability(double x)  {
        double ret;
        if (x <= 0.0) {
            ret = 0.0;
        } else {
            ret = 1.0 - FastMath.exp(-x / mean);
        }
        return ret;
    }

    
    @Override
    public double inverseCumulativeProbability(double p) throws OutOfRangeException {
        double ret;

        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0.0, 1.0);
        } else if (p == 1.0) {
            ret = Double.POSITIVE_INFINITY;
        } else {
            ret = -mean * FastMath.log(1.0 - p);
        }

        return ret;
    }

    
    @Override
    public double sample() {
        // Step 1:
        double a = 0;
        double u = random.nextDouble();

        // Step 2 and 3:
        while (u < 0.5) {
            a += EXPONENTIAL_SA_QI[0];
            u *= 2;
        }

        // Step 4 (now u >= 0.5):
        u += u - 1;

        // Step 5:
        if (u <= EXPONENTIAL_SA_QI[0]) {
            return mean * (a + u);
        }

        // Step 6:
        int i = 0; // Should be 1, be we iterate before it in while using 0
        double u2 = random.nextDouble();
        double umin = u2;

        // Step 7 and 8:
        do {
            ++i;
            u2 = random.nextDouble();

            if (u2 < umin) {
                umin = u2;
            }

            // Step 8:
        } while (u > EXPONENTIAL_SA_QI[i]); // Ensured to exit since EXPONENTIAL_SA_QI[MAX] = 1

        return mean * (a + umin * EXPONENTIAL_SA_QI[0]);
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        return getMean();
    }

    
    public double getNumericalVariance() {
        final double m = getMean();
        return m * m;
    }

    
    public double getSupportLowerBound() {
        return 0;
    }

    
    public double getSupportUpperBound() {
        return Double.POSITIVE_INFINITY;
    }

    
    public boolean isSupportLowerBoundInclusive() {
        return true;
    }

    
    public boolean isSupportUpperBoundInclusive() {
        return false;
    }

    
    public boolean isSupportConnected() {
        return true;
    }
}
