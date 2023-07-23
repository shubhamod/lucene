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

import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;


public class UniformRealDistribution extends AbstractRealDistribution {
    
    @Deprecated
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = 20120109L;
    
    private final double lower;
    
    private final double upper;

    
    public UniformRealDistribution() {
        this(0, 1);
    }

    
    public UniformRealDistribution(double lower, double upper)
        throws NumberIsTooLargeException {
        this(new Well19937c(), lower, upper);
    }

    
    @Deprecated
    public UniformRealDistribution(double lower, double upper, double inverseCumAccuracy)
        throws NumberIsTooLargeException {
        this(new Well19937c(), lower, upper);
    }

    
    @Deprecated
    public UniformRealDistribution(RandomGenerator rng,
                                   double lower,
                                   double upper,
                                   double inverseCumAccuracy){
        this(rng, lower, upper);
    }

    
    public UniformRealDistribution(RandomGenerator rng,
                                   double lower,
                                   double upper)
        throws NumberIsTooLargeException {
        super(rng);
        if (lower >= upper) {
            throw new NumberIsTooLargeException(
                            LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND,
                            lower, upper, false);
        }

        this.lower = lower;
        this.upper = upper;
    }

    
    public double density(double x) {
        if (x < lower || x > upper) {
            return 0.0;
        }
        return 1 / (upper - lower);
    }

    
    public double cumulativeProbability(double x)  {
        if (x <= lower) {
            return 0;
        }
        if (x >= upper) {
            return 1;
        }
        return (x - lower) / (upper - lower);
    }

    
    @Override
    public double inverseCumulativeProbability(final double p)
            throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }
        return p * (upper - lower) + lower;
    }

    
    public double getNumericalMean() {
        return 0.5 * (lower + upper);
    }

    
    public double getNumericalVariance() {
        double ul = upper - lower;
        return ul * ul / 12;
    }

    
    public double getSupportLowerBound() {
        return lower;
    }

    
    public double getSupportUpperBound() {
        return upper;
    }

    
    public boolean isSupportLowerBoundInclusive() {
        return true;
    }

    
    public boolean isSupportUpperBoundInclusive() {
        return true;
    }

    
    public boolean isSupportConnected() {
        return true;
    }

    
    @Override
    public double sample()  {
        final double u = random.nextDouble();
        return u * upper + (1 - u) * lower;
    }
}
