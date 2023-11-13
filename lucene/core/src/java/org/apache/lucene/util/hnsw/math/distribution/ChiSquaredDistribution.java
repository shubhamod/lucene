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

import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;


public class ChiSquaredDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = -8352658048349159782L;
    
    private final GammaDistribution gamma;
    
    private final double solverAbsoluteAccuracy;

    
    public ChiSquaredDistribution(double degreesOfFreedom) {
        this(degreesOfFreedom, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public ChiSquaredDistribution(double degreesOfFreedom,
                                  double inverseCumAccuracy) {
        this(new Well19937c(), degreesOfFreedom, inverseCumAccuracy);
    }

    
    public ChiSquaredDistribution(RandomGenerator rng, double degreesOfFreedom) {
        this(rng, degreesOfFreedom, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public ChiSquaredDistribution(RandomGenerator rng,
                                  double degreesOfFreedom,
                                  double inverseCumAccuracy) {
        super(rng);

        gamma = new GammaDistribution(degreesOfFreedom / 2, 2);
        solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    
    public double getDegreesOfFreedom() {
        return gamma.getShape() * 2.0;
    }

    
    public double density(double x) {
        return gamma.density(x);
    }

    
    @Override
    public double logDensity(double x) {
        return gamma.logDensity(x);
    }

    
    public double cumulativeProbability(double x)  {
        return gamma.cumulativeProbability(x);
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        return getDegreesOfFreedom();
    }

    
    public double getNumericalVariance() {
        return 2 * getDegreesOfFreedom();
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
