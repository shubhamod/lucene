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
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class ParetoDistribution extends AbstractRealDistribution {

    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;

    
    private static final long serialVersionUID = 20130424;

    
    private final double scale;

    
    private final double shape;

    
    private final double solverAbsoluteAccuracy;

    
    public ParetoDistribution() {
        this(1, 1);
    }

    
    public ParetoDistribution(double scale, double shape)
        throws NotStrictlyPositiveException {
        this(scale, shape, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public ParetoDistribution(double scale, double shape, double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        this(new Well19937c(), scale, shape, inverseCumAccuracy);
    }

    
    public ParetoDistribution(RandomGenerator rng, double scale, double shape)
        throws NotStrictlyPositiveException {
        this(rng, scale, shape, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public ParetoDistribution(RandomGenerator rng,
                              double scale,
                              double shape,
                              double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        super(rng);

        if (scale <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SCALE, scale);
        }

        if (shape <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SHAPE, shape);
        }

        this.scale = scale;
        this.shape = shape;
        this.solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    
    public double getScale() {
        return scale;
    }

    
    public double getShape() {
        return shape;
    }

    
    public double density(double x) {
        if (x < scale) {
            return 0;
        }
        return FastMath.pow(scale, shape) / FastMath.pow(x, shape + 1) * shape;
    }

    
    @Override
    public double logDensity(double x) {
        if (x < scale) {
            return Double.NEGATIVE_INFINITY;
        }
        return FastMath.log(scale) * shape - FastMath.log(x) * (shape + 1) + FastMath.log(shape);
    }

    
    public double cumulativeProbability(double x)  {
        if (x <= scale) {
            return 0;
        }
        return 1 - FastMath.pow(scale / x, shape);
    }

    
    @Override
    @Deprecated
    public double cumulativeProbability(double x0, double x1)
        throws NumberIsTooLargeException {
        return probability(x0, x1);
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        if (shape <= 1) {
            return Double.POSITIVE_INFINITY;
        }
        return shape * scale / (shape - 1);
    }

    
    public double getNumericalVariance() {
        if (shape <= 2) {
            return Double.POSITIVE_INFINITY;
        }
        double s = shape - 1;
        return scale * scale * shape / (s * s) / (shape - 2);
    }

    
    public double getSupportLowerBound() {
        return scale;
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

    
    @Override
    public double sample()  {
        final double n = random.nextDouble();
        return scale / FastMath.pow(n, 1 / shape);
    }
}
