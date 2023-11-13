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


public class CauchyDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = 8589540077390120676L;
    
    private final double median;
    
    private final double scale;
    
    private final double solverAbsoluteAccuracy;

    
    public CauchyDistribution() {
        this(0, 1);
    }

    
    public CauchyDistribution(double median, double scale) {
        this(median, scale, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public CauchyDistribution(double median, double scale,
                              double inverseCumAccuracy) {
        this(new Well19937c(), median, scale, inverseCumAccuracy);
    }

    
    public CauchyDistribution(RandomGenerator rng, double median, double scale) {
        this(rng, median, scale, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public CauchyDistribution(RandomGenerator rng,
                              double median,
                              double scale,
                              double inverseCumAccuracy) {
        super(rng);
        if (scale <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SCALE, scale);
        }
        this.scale = scale;
        this.median = median;
        solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    
    public double cumulativeProbability(double x) {
        return 0.5 + (FastMath.atan((x - median) / scale) / FastMath.PI);
    }

    
    public double getMedian() {
        return median;
    }

    
    public double getScale() {
        return scale;
    }

    
    public double density(double x) {
        final double dev = x - median;
        return (1 / FastMath.PI) * (scale / (dev * dev + scale * scale));
    }

    
    @Override
    public double inverseCumulativeProbability(double p) throws OutOfRangeException {
        double ret;
        if (p < 0 || p > 1) {
            throw new OutOfRangeException(p, 0, 1);
        } else if (p == 0) {
            ret = Double.NEGATIVE_INFINITY;
        } else  if (p == 1) {
            ret = Double.POSITIVE_INFINITY;
        } else {
            ret = median + scale * FastMath.tan(FastMath.PI * (p - .5));
        }
        return ret;
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        return Double.NaN;
    }

    
    public double getNumericalVariance() {
        return Double.NaN;
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
