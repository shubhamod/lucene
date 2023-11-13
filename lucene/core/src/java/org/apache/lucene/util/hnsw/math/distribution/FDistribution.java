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
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class FDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = -8516354193418641566L;
    
    private final double numeratorDegreesOfFreedom;
    
    private final double denominatorDegreesOfFreedom;
    
    private final double solverAbsoluteAccuracy;
    
    private double numericalVariance = Double.NaN;
    
    private boolean numericalVarianceIsCalculated = false;

    
    public FDistribution(double numeratorDegreesOfFreedom,
                         double denominatorDegreesOfFreedom)
        throws NotStrictlyPositiveException {
        this(numeratorDegreesOfFreedom, denominatorDegreesOfFreedom,
             DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public FDistribution(double numeratorDegreesOfFreedom,
                         double denominatorDegreesOfFreedom,
                         double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        this(new Well19937c(), numeratorDegreesOfFreedom,
             denominatorDegreesOfFreedom, inverseCumAccuracy);
    }

    
    public FDistribution(RandomGenerator rng,
                         double numeratorDegreesOfFreedom,
                         double denominatorDegreesOfFreedom)
        throws NotStrictlyPositiveException {
        this(rng, numeratorDegreesOfFreedom, denominatorDegreesOfFreedom, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public FDistribution(RandomGenerator rng,
                         double numeratorDegreesOfFreedom,
                         double denominatorDegreesOfFreedom,
                         double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        super(rng);

        if (numeratorDegreesOfFreedom <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.DEGREES_OF_FREEDOM,
                                                   numeratorDegreesOfFreedom);
        }
        if (denominatorDegreesOfFreedom <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.DEGREES_OF_FREEDOM,
                                                   denominatorDegreesOfFreedom);
        }
        this.numeratorDegreesOfFreedom = numeratorDegreesOfFreedom;
        this.denominatorDegreesOfFreedom = denominatorDegreesOfFreedom;
        solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    
    public double density(double x) {
        return FastMath.exp(logDensity(x));
    }

    
    @Override
    public double logDensity(double x) {
        final double nhalf = numeratorDegreesOfFreedom / 2;
        final double mhalf = denominatorDegreesOfFreedom / 2;
        final double logx = FastMath.log(x);
        final double logn = FastMath.log(numeratorDegreesOfFreedom);
        final double logm = FastMath.log(denominatorDegreesOfFreedom);
        final double lognxm = FastMath.log(numeratorDegreesOfFreedom * x +
                denominatorDegreesOfFreedom);
        return nhalf * logn + nhalf * logx - logx +
               mhalf * logm - nhalf * lognxm - mhalf * lognxm -
               Beta.logBeta(nhalf, mhalf);
    }

    
    public double cumulativeProbability(double x)  {
        double ret;
        if (x <= 0) {
            ret = 0;
        } else {
            double n = numeratorDegreesOfFreedom;
            double m = denominatorDegreesOfFreedom;

            ret = Beta.regularizedBeta((n * x) / (m + n * x),
                0.5 * n,
                0.5 * m);
        }
        return ret;
    }

    
    public double getNumeratorDegreesOfFreedom() {
        return numeratorDegreesOfFreedom;
    }

    
    public double getDenominatorDegreesOfFreedom() {
        return denominatorDegreesOfFreedom;
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        final double denominatorDF = getDenominatorDegreesOfFreedom();

        if (denominatorDF > 2) {
            return denominatorDF / (denominatorDF - 2);
        }

        return Double.NaN;
    }

    
    public double getNumericalVariance() {
        if (!numericalVarianceIsCalculated) {
            numericalVariance = calculateNumericalVariance();
            numericalVarianceIsCalculated = true;
        }
        return numericalVariance;
    }

    
    protected double calculateNumericalVariance() {
        final double denominatorDF = getDenominatorDegreesOfFreedom();

        if (denominatorDF > 4) {
            final double numeratorDF = getNumeratorDegreesOfFreedom();
            final double denomDFMinusTwo = denominatorDF - 2;

            return ( 2 * (denominatorDF * denominatorDF) * (numeratorDF + denominatorDF - 2) ) /
                   ( (numeratorDF * (denomDFMinusTwo * denomDFMinusTwo) * (denominatorDF - 4)) );
        }

        return Double.NaN;
    }

    
    public double getSupportLowerBound() {
        return 0;
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
