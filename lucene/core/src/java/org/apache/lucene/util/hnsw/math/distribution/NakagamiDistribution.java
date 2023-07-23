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
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.special.Gamma;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class NakagamiDistribution extends AbstractRealDistribution {

    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;

    
    private static final long serialVersionUID = 20141003;

    
    private final double mu;
    
    private final double omega;
    
    private final double inverseAbsoluteAccuracy;

    
    public NakagamiDistribution(double mu, double omega) {
        this(mu, omega, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public NakagamiDistribution(double mu, double omega, double inverseAbsoluteAccuracy) {
        this(new Well19937c(), mu, omega, inverseAbsoluteAccuracy);
    }

    
    public NakagamiDistribution(RandomGenerator rng, double mu, double omega, double inverseAbsoluteAccuracy) {
        super(rng);

        if (mu < 0.5) {
            throw new NumberIsTooSmallException(mu, 0.5, true);
        }
        if (omega <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NOT_POSITIVE_SCALE, omega);
        }

        this.mu = mu;
        this.omega = omega;
        this.inverseAbsoluteAccuracy = inverseAbsoluteAccuracy;
    }

    
    public double getShape() {
        return mu;
    }

    
    public double getScale() {
        return omega;
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return inverseAbsoluteAccuracy;
    }

    
    public double density(double x) {
        if (x <= 0) {
            return 0.0;
        }
        return 2.0 * FastMath.pow(mu, mu) / (Gamma.gamma(mu) * FastMath.pow(omega, mu)) *
                     FastMath.pow(x, 2 * mu - 1) * FastMath.exp(-mu * x * x / omega);
    }

    
    public double cumulativeProbability(double x) {
        return Gamma.regularizedGammaP(mu, mu * x * x / omega);
    }

    
    public double getNumericalMean() {
        return Gamma.gamma(mu + 0.5) / Gamma.gamma(mu) * FastMath.sqrt(omega / mu);
    }

    
    public double getNumericalVariance() {
        double v = Gamma.gamma(mu + 0.5) / Gamma.gamma(mu);
        return omega * (1 - 1 / mu * v * v);
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
