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
import org.apache.lucene.util.hnsw.math.special.Erf;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class LogNormalDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;

    
    private static final long serialVersionUID = 20120112;

    
    private static final double SQRT2PI = FastMath.sqrt(2 * FastMath.PI);

    
    private static final double SQRT2 = FastMath.sqrt(2.0);

    
    private final double scale;

    
    private final double shape;
    
    private final double logShapePlusHalfLog2Pi;

    
    private final double solverAbsoluteAccuracy;

    
    public LogNormalDistribution() {
        this(0, 1);
    }

    
    public LogNormalDistribution(double scale, double shape)
        throws NotStrictlyPositiveException {
        this(scale, shape, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public LogNormalDistribution(double scale, double shape, double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        this(new Well19937c(), scale, shape, inverseCumAccuracy);
    }

    
    public LogNormalDistribution(RandomGenerator rng, double scale, double shape)
        throws NotStrictlyPositiveException {
        this(rng, scale, shape, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public LogNormalDistribution(RandomGenerator rng,
                                 double scale,
                                 double shape,
                                 double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        super(rng);

        if (shape <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SHAPE, shape);
        }

        this.scale = scale;
        this.shape = shape;
        this.logShapePlusHalfLog2Pi = FastMath.log(shape) + 0.5 * FastMath.log(2 * FastMath.PI);
        this.solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    
    public double getScale() {
        return scale;
    }

    
    public double getShape() {
        return shape;
    }

    
    public double density(double x) {
        if (x <= 0) {
            return 0;
        }
        final double x0 = FastMath.log(x) - scale;
        final double x1 = x0 / shape;
        return FastMath.exp(-0.5 * x1 * x1) / (shape * SQRT2PI * x);
    }

    
    @Override
    public double logDensity(double x) {
        if (x <= 0) {
            return Double.NEGATIVE_INFINITY;
        }
        final double logX = FastMath.log(x);
        final double x0 = logX - scale;
        final double x1 = x0 / shape;
        return -0.5 * x1 * x1 - (logShapePlusHalfLog2Pi + logX);
    }

    
    public double cumulativeProbability(double x)  {
        if (x <= 0) {
            return 0;
        }
        final double dev = FastMath.log(x) - scale;
        if (FastMath.abs(dev) > 40 * shape) {
            return dev < 0 ? 0.0d : 1.0d;
        }
        return 0.5 + 0.5 * Erf.erf(dev / (shape * SQRT2));
    }

    
    @Override@Deprecated
    public double cumulativeProbability(double x0, double x1)
        throws NumberIsTooLargeException {
        return probability(x0, x1);
    }

    
    @Override
    public double probability(double x0,
                              double x1)
        throws NumberIsTooLargeException {
        if (x0 > x1) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_ENDPOINT_ABOVE_UPPER_ENDPOINT,
                                                x0, x1, true);
        }
        if (x0 <= 0 || x1 <= 0) {
            return super.probability(x0, x1);
        }
        final double denom = shape * SQRT2;
        final double v0 = (FastMath.log(x0) - scale) / denom;
        final double v1 = (FastMath.log(x1) - scale) / denom;
        return 0.5 * Erf.erf(v0, v1);
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        double s = shape;
        return FastMath.exp(scale + (s * s / 2));
    }

    
    public double getNumericalVariance() {
        final double s = shape;
        final double ss = s * s;
        return (FastMath.expm1(ss)) * FastMath.exp(2 * scale + ss);
    }

    
    public double getSupportLowerBound() {
        return 0;
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
        final double n = random.nextGaussian();
        return FastMath.exp(scale + shape * n);
    }
}
