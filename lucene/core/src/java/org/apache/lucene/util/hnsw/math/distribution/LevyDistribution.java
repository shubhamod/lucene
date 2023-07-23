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

import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.special.Erf;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class LevyDistribution extends AbstractRealDistribution {

    
    private static final long serialVersionUID = 20130314L;

    
    private final double mu;

    
    private final double c;  // Setting this to 1 returns a cumProb of 1.0

    
    private final double halfC;

    
    public LevyDistribution(final double mu, final double c) {
        this(new Well19937c(), mu, c);
    }

    
    public LevyDistribution(final RandomGenerator rng, final double mu, final double c) {
        super(rng);
        this.mu    = mu;
        this.c     = c;
        this.halfC = 0.5 * c;
    }

    
    public double density(final double x) {
        if (x < mu) {
            return Double.NaN;
        }

        final double delta = x - mu;
        final double f     = halfC / delta;
        return FastMath.sqrt(f / FastMath.PI) * FastMath.exp(-f) /delta;
    }

    
    @Override
    public double logDensity(double x) {
        if (x < mu) {
            return Double.NaN;
        }

        final double delta = x - mu;
        final double f     = halfC / delta;
        return 0.5 * FastMath.log(f / FastMath.PI) - f - FastMath.log(delta);
    }

    
    public double cumulativeProbability(final double x) {
        if (x < mu) {
            return Double.NaN;
        }
        return Erf.erfc(FastMath.sqrt(halfC / (x - mu)));
    }

    
    @Override
    public double inverseCumulativeProbability(final double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }
        final double t = Erf.erfcInv(p);
        return mu + halfC / (t * t);
    }

    
    public double getScale() {
        return c;
    }

    
    public double getLocation() {
        return mu;
    }

    
    public double getNumericalMean() {
        return Double.POSITIVE_INFINITY;
    }

    
    public double getNumericalVariance() {
        return Double.POSITIVE_INFINITY;
    }

    
    public double getSupportLowerBound() {
        return mu;
    }

    
    public double getSupportUpperBound() {
        return Double.POSITIVE_INFINITY;
    }

    
    public boolean isSupportLowerBoundInclusive() {
        // there is a division by x-mu in the computation, so density
        // is not finite at lower bound, bound must be excluded
        return false;
    }

    
    public boolean isSupportUpperBoundInclusive() {
        // upper bound is infinite, so it must be excluded
        return false;
    }

    
    public boolean isSupportConnected() {
        return true;
    }

}
