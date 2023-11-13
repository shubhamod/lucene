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

import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.special.Beta;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class BinomialDistribution extends AbstractIntegerDistribution {
    
    private static final long serialVersionUID = 6751309484392813623L;
    
    private final int numberOfTrials;
    
    private final double probabilityOfSuccess;

    
    public BinomialDistribution(int trials, double p) {
        this(new Well19937c(), trials, p);
    }

    
    public BinomialDistribution(RandomGenerator rng,
                                int trials,
                                double p) {
        super(rng);

        if (trials < 0) {
            throw new NotPositiveException(LocalizedFormats.NUMBER_OF_TRIALS,
                                           trials);
        }
        if (p < 0 || p > 1) {
            throw new OutOfRangeException(p, 0, 1);
        }

        probabilityOfSuccess = p;
        numberOfTrials = trials;
    }

    
    public int getNumberOfTrials() {
        return numberOfTrials;
    }

    
    public double getProbabilityOfSuccess() {
        return probabilityOfSuccess;
    }

    
    public double probability(int x) {
        final double logProbability = logProbability(x);
        return logProbability == Double.NEGATIVE_INFINITY ? 0 : FastMath.exp(logProbability);
    }

    
    @Override
    public double logProbability(int x) {
        if (numberOfTrials == 0) {
            return (x == 0) ? 0. : Double.NEGATIVE_INFINITY;
        }
        double ret;
        if (x < 0 || x > numberOfTrials) {
            ret = Double.NEGATIVE_INFINITY;
        } else {
            ret = SaddlePointExpansion.logBinomialProbability(x,
                    numberOfTrials, probabilityOfSuccess,
                    1.0 - probabilityOfSuccess);
        }
        return ret;
    }

    
    public double cumulativeProbability(int x) {
        double ret;
        if (x < 0) {
            ret = 0.0;
        } else if (x >= numberOfTrials) {
            ret = 1.0;
        } else {
            ret = 1.0 - Beta.regularizedBeta(probabilityOfSuccess,
                    x + 1.0, numberOfTrials - x);
        }
        return ret;
    }

    
    public double getNumericalMean() {
        return numberOfTrials * probabilityOfSuccess;
    }

    
    public double getNumericalVariance() {
        final double p = probabilityOfSuccess;
        return numberOfTrials * p * (1 - p);
    }

    
    public int getSupportLowerBound() {
        return probabilityOfSuccess < 1.0 ? 0 : numberOfTrials;
    }

    
    public int getSupportUpperBound() {
        return probabilityOfSuccess > 0.0 ? numberOfTrials : 0;
    }

    
    public boolean isSupportConnected() {
        return true;
    }
}
