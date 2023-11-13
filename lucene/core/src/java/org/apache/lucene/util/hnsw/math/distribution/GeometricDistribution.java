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
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class GeometricDistribution extends AbstractIntegerDistribution {

    
    private static final long serialVersionUID = 20130507L;
    
    private final double probabilityOfSuccess;
    
    private final double logProbabilityOfSuccess;
    
    private final double log1mProbabilityOfSuccess;

    
    public GeometricDistribution(double p) {
        this(new Well19937c(), p);
    }

    
    public GeometricDistribution(RandomGenerator rng, double p) {
        super(rng);

        if (p <= 0 || p > 1) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_RANGE_LEFT, p, 0, 1);
        }

        probabilityOfSuccess = p;
        logProbabilityOfSuccess = FastMath.log(p);
        log1mProbabilityOfSuccess = FastMath.log1p(-p);
    }

    
    public double getProbabilityOfSuccess() {
        return probabilityOfSuccess;
    }

    
    public double probability(int x) {
        if (x < 0) {
            return 0.0;
        } else {
            return FastMath.exp(log1mProbabilityOfSuccess * x) * probabilityOfSuccess;
        }
    }

    
    @Override
    public double logProbability(int x) {
        if (x < 0) {
            return Double.NEGATIVE_INFINITY;
        } else {
            return x * log1mProbabilityOfSuccess + logProbabilityOfSuccess;
        }
    }

    
    public double cumulativeProbability(int x) {
        if (x < 0) {
            return 0.0;
        } else {
            return -FastMath.expm1(log1mProbabilityOfSuccess * (x + 1));
        }
    }

    
    public double getNumericalMean() {
        return (1 - probabilityOfSuccess) / probabilityOfSuccess;
    }

    
    public double getNumericalVariance() {
        return (1 - probabilityOfSuccess) / (probabilityOfSuccess * probabilityOfSuccess);
    }

    
    public int getSupportLowerBound() {
        return 0;
    }

    
    public int getSupportUpperBound() {
        return Integer.MAX_VALUE;
    }

    
    public boolean isSupportConnected() {
        return true;
    }

    
    @Override
    public int inverseCumulativeProbability(double p) throws OutOfRangeException {
        if (p < 0 || p > 1) {
            throw new OutOfRangeException(p, 0, 1);
        }
        if (p == 1) {
            return Integer.MAX_VALUE;
        }
        if (p == 0) {
            return 0;
        }
        return Math.max(0, (int) Math.ceil(FastMath.log1p(-p)/log1mProbabilityOfSuccess-1));
    }
}
