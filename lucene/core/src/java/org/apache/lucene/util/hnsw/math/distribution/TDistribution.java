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
import org.apache.lucene.util.hnsw.math.special.Beta;
import org.apache.lucene.util.hnsw.math.special.Gamma;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class TDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = -5852615386664158222L;
    
    private final double degreesOfFreedom;
    
    private final double solverAbsoluteAccuracy;
    
    private final double factor;

    
    public TDistribution(double degreesOfFreedom)
        throws NotStrictlyPositiveException {
        this(degreesOfFreedom, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public TDistribution(double degreesOfFreedom, double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        this(new Well19937c(), degreesOfFreedom, inverseCumAccuracy);
    }

    
    public TDistribution(RandomGenerator rng, double degreesOfFreedom)
        throws NotStrictlyPositiveException {
        this(rng, degreesOfFreedom, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public TDistribution(RandomGenerator rng,
                         double degreesOfFreedom,
                         double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        super(rng);

        if (degreesOfFreedom <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.DEGREES_OF_FREEDOM,
                                                   degreesOfFreedom);
        }
        this.degreesOfFreedom = degreesOfFreedom;
        solverAbsoluteAccuracy = inverseCumAccuracy;

        final double n = degreesOfFreedom;
        final double nPlus1Over2 = (n + 1) / 2;
        factor = Gamma.logGamma(nPlus1Over2) -
                 0.5 * (FastMath.log(FastMath.PI) + FastMath.log(n)) -
                 Gamma.logGamma(n / 2);
    }

    
    public double getDegreesOfFreedom() {
        return degreesOfFreedom;
    }

    
    public double density(double x) {
        return FastMath.exp(logDensity(x));
    }

    
    @Override
    public double logDensity(double x) {
        final double n = degreesOfFreedom;
        final double nPlus1Over2 = (n + 1) / 2;
        return factor - nPlus1Over2 * FastMath.log(1 + x * x / n);
    }

    
    public double cumulativeProbability(double x) {
        double ret;
        if (x == 0) {
            ret = 0.5;
        } else {
            double t =
                Beta.regularizedBeta(
                    degreesOfFreedom / (degreesOfFreedom + (x * x)),
                    0.5 * degreesOfFreedom,
                    0.5);
            if (x < 0.0) {
                ret = 0.5 * t;
            } else {
                ret = 1.0 - 0.5 * t;
            }
        }

        return ret;
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        final double df = getDegreesOfFreedom();

        if (df > 1) {
            return 0;
        }

        return Double.NaN;
    }

    
    public double getNumericalVariance() {
        final double df = getDegreesOfFreedom();

        if (df > 2) {
            return df / (df - 2);
        }

        if (df > 1 && df <= 2) {
            return Double.POSITIVE_INFINITY;
        }

        return Double.NaN;
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
}
