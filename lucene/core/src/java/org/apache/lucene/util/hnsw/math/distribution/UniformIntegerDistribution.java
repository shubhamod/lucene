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
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;


public class UniformIntegerDistribution extends AbstractIntegerDistribution {
    
    private static final long serialVersionUID = 20120109L;
    
    private final int lower;
    
    private final int upper;

    
    public UniformIntegerDistribution(int lower, int upper)
        throws NumberIsTooLargeException {
        this(new Well19937c(), lower, upper);
    }

    
    public UniformIntegerDistribution(RandomGenerator rng,
                                      int lower,
                                      int upper)
        throws NumberIsTooLargeException {
        super(rng);

        if (lower > upper) {
            throw new NumberIsTooLargeException(
                            LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND,
                            lower, upper, true);
        }
        this.lower = lower;
        this.upper = upper;
    }

    
    public double probability(int x) {
        if (x < lower || x > upper) {
            return 0;
        }
        return 1.0 / (upper - lower + 1);
    }

    
    public double cumulativeProbability(int x) {
        if (x < lower) {
            return 0;
        }
        if (x > upper) {
            return 1;
        }
        return (x - lower + 1.0) / (upper - lower + 1.0);
    }

    
    public double getNumericalMean() {
        return 0.5 * (lower + upper);
    }

    
    public double getNumericalVariance() {
        double n = upper - lower + 1;
        return (n * n - 1) / 12.0;
    }

    
    public int getSupportLowerBound() {
        return lower;
    }

    
    public int getSupportUpperBound() {
        return upper;
    }

    
    public boolean isSupportConnected() {
        return true;
    }

    
    @Override
    public int sample() {
        final int max = (upper - lower) + 1;
        if (max <= 0) {
            // The range is too wide to fit in a positive int (larger
            // than 2^31); as it covers more than half the integer range,
            // we use a simple rejection method.
            while (true) {
                final int r = random.nextInt();
                if (r >= lower &&
                    r <= upper) {
                    return r;
                }
            }
        } else {
            // We can shift the range and directly generate a positive int.
            return lower + random.nextInt(max);
        }
    }
}
