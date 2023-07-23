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


public class ConstantRealDistribution extends AbstractRealDistribution {

    
    private static final long serialVersionUID = -4157745166772046273L;

    
    private final double value;

    
    public ConstantRealDistribution(double value) {
        super(null);  // Avoid creating RandomGenerator
        this.value = value;
    }

    
    public double density(double x) {
        return x == value ? 1 : 0;
    }

    
    public double cumulativeProbability(double x)  {
        return x < value ? 0 : 1;
    }

    
    @Override
    public double inverseCumulativeProbability(final double p)
            throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }
        return value;
    }

    
    public double getNumericalMean() {
        return value;
    }

    
    public double getNumericalVariance() {
        return 0;
    }

    
    public double getSupportLowerBound() {
        return value;
    }

    
    public double getSupportUpperBound() {
        return value;
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
        return value;
    }

    
    @Override
    public void reseedRandomGenerator(long seed) {}
}
