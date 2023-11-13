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
import org.apache.lucene.util.hnsw.math.special.Beta;
import org.apache.lucene.util.hnsw.math.util.CombinatoricsUtils;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class PascalDistribution extends AbstractIntegerDistribution {
    
    private static final long serialVersionUID = 6751309484392813623L;
    
    private final int numberOfSuccesses;
    
    private final double probabilityOfSuccess;
    
    private final double logProbabilityOfSuccess;
    
    private final double log1mProbabilityOfSuccess;

    
    public PascalDistribution(int r, double p)
        throws NotStrictlyPositiveException, OutOfRangeException {
        this(new Well19937c(), r, p);
    }

    
    public PascalDistribution(RandomGenerator rng,
                              int r,
                              double p)
        throws NotStrictlyPositiveException, OutOfRangeException {
        super(rng);

        if (r <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_SUCCESSES,
                                                   r);
        }
        if (p < 0 || p > 1) {
            throw new OutOfRangeException(p, 0, 1);
        }

        numberOfSuccesses = r;
        probabilityOfSuccess = p;
        logProbabilityOfSuccess = FastMath.log(p);
        log1mProbabilityOfSuccess = FastMath.log1p(-p);
    }

    
    public int getNumberOfSuccesses() {
        return numberOfSuccesses;
    }

    
    public double getProbabilityOfSuccess() {
        return probabilityOfSuccess;
    }

    
    public double probability(int x) {
        double ret;
        if (x < 0) {
            ret = 0.0;
        } else {
            ret = CombinatoricsUtils.binomialCoefficientDouble(x +
                  numberOfSuccesses - 1, numberOfSuccesses - 1) *
                  FastMath.pow(probabilityOfSuccess, numberOfSuccesses) *
                  FastMath.pow(1.0 - probabilityOfSuccess, x);
        }
        return ret;
    }

    
    @Override
    public double logProbability(int x) {
        double ret;
        if (x < 0) {
            ret = Double.NEGATIVE_INFINITY;
        } else {
            ret = CombinatoricsUtils.binomialCoefficientLog(x +
                  numberOfSuccesses - 1, numberOfSuccesses - 1) +
                  logProbabilityOfSuccess * numberOfSuccesses +
                  log1mProbabilityOfSuccess * x;
        }
        return ret;
    }

    
    public double cumulativeProbability(int x) {
        double ret;
        if (x < 0) {
            ret = 0.0;
        } else {
            ret = Beta.regularizedBeta(probabilityOfSuccess,
                    numberOfSuccesses, x + 1.0);
        }
        return ret;
    }

    
    public double getNumericalMean() {
        final double p = getProbabilityOfSuccess();
        final double r = getNumberOfSuccesses();
        return (r * (1 - p)) / p;
    }

    
    public double getNumericalVariance() {
        final double p = getProbabilityOfSuccess();
        final double r = getNumberOfSuccesses();
        return r * (1 - p) / (p * p);
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
}
