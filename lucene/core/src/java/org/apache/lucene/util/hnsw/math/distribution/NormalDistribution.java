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
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.special.Erf;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class NormalDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = 8589540077390120676L;
    
    private static final double SQRT2 = FastMath.sqrt(2.0);
    
    private final double mean;
    
    private final double standardDeviation;
    
    private final double logStandardDeviationPlusHalfLog2Pi;
    
    private final double solverAbsoluteAccuracy;

    
    public NormalDistribution() {
        this(0, 1);
    }

    
    public NormalDistribution(double mean, double sd)
        throws NotStrictlyPositiveException {
        this(mean, sd, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public NormalDistribution(double mean, double sd, double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        this(new Well19937c(), mean, sd, inverseCumAccuracy);
    }

    
    public NormalDistribution(RandomGenerator rng, double mean, double sd)
        throws NotStrictlyPositiveException {
        this(rng, mean, sd, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public NormalDistribution(RandomGenerator rng,
                              double mean,
                              double sd,
                              double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        super(rng);

        if (sd <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.STANDARD_DEVIATION, sd);
        }

        this.mean = mean;
        standardDeviation = sd;
        logStandardDeviationPlusHalfLog2Pi = FastMath.log(sd) + 0.5 * FastMath.log(2 * FastMath.PI);
        solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    
    public double getMean() {
        return mean;
    }

    
    public double getStandardDeviation() {
        return standardDeviation;
    }

    
    public double density(double x) {
        return FastMath.exp(logDensity(x));
    }

    
    @Override
    public double logDensity(double x) {
        final double x0 = x - mean;
        final double x1 = x0 / standardDeviation;
        return -0.5 * x1 * x1 - logStandardDeviationPlusHalfLog2Pi;
    }

    
    public double cumulativeProbability(double x)  {
        final double dev = x - mean;
        if (FastMath.abs(dev) > 40 * standardDeviation) {
            return dev < 0 ? 0.0d : 1.0d;
        }
        return 0.5 * Erf.erfc(-dev / (standardDeviation * SQRT2));
    }

    
    @Override
    public double inverseCumulativeProbability(final double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }
        return mean + standardDeviation * SQRT2 * Erf.erfInv(2 * p - 1);
    }

    
    @Override@Deprecated
    public double cumulativeProbability(double x0, double x1)
        throws NumberIsTooLargeException {
        return probability(x0, x1);
    }

    
    @Override
    public double probability(double x0,
                              double x1)
        throws NumberIsTooLargeException {
        if (x0 > x1) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_ENDPOINT_ABOVE_UPPER_ENDPOINT,
                                                x0, x1, true);
        }
        final double denom = standardDeviation * SQRT2;
        final double v0 = (x0 - mean) / denom;
        final double v1 = (x1 - mean) / denom;
        return 0.5 * Erf.erf(v0, v1);
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        return getMean();
    }

    
    public double getNumericalVariance() {
        final double s = getStandardDeviation();
        return s * s;
    }

    
    public double getSupportLowerBound() {
        return Double.NEGATIVE_INFINITY;
    }

    
    public double getSupportUpperBound() {
        return Double.POSITIVE_INFINITY;
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
    public double sample()  {
        return standardDeviation * random.nextGaussian() + mean;
    }
}
