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
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class LaplaceDistribution extends AbstractRealDistribution {

    
    private static final long serialVersionUID = 20141003;

    
    private final double mu;
    
    private final double beta;

    
    public LaplaceDistribution(double mu, double beta) {
        this(new Well19937c(), mu, beta);
    }

    
    public LaplaceDistribution(RandomGenerator rng, double mu, double beta) {
        super(rng);

        if (beta <= 0.0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NOT_POSITIVE_SCALE, beta);
        }

        this.mu = mu;
        this.beta = beta;
    }

    
    public double getLocation() {
        return mu;
    }

    
    public double getScale() {
        return beta;
    }

    
    public double density(double x) {
        return FastMath.exp(-FastMath.abs(x - mu) / beta) / (2.0 * beta);
    }

    
    public double cumulativeProbability(double x) {
        if (x <= mu) {
            return FastMath.exp((x - mu) / beta) / 2.0;
        } else {
            return 1.0 - FastMath.exp((mu - x) / beta) / 2.0;
        }
    }

    
    @Override
    public double inverseCumulativeProbability(double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0.0, 1.0);
        } else if (p == 0) {
            return Double.NEGATIVE_INFINITY;
        } else if (p == 1) {
            return Double.POSITIVE_INFINITY;
        }
        double x = (p > 0.5) ? -Math.log(2.0 - 2.0 * p) : Math.log(2.0 * p);
        return mu + beta * x;
    }

    
    public double getNumericalMean() {
        return mu;
    }

    
    public double getNumericalVariance() {
        return 2.0 * beta * beta;
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
