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
import org.apache.lucene.util.hnsw.math.special.Gamma;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class WeibullDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = 8589540077390120676L;
    
    private final double shape;
    
    private final double scale;
    
    private final double solverAbsoluteAccuracy;
    
    private double numericalMean = Double.NaN;
    
    private boolean numericalMeanIsCalculated = false;
    
    private double numericalVariance = Double.NaN;
    
    private boolean numericalVarianceIsCalculated = false;

    
    public WeibullDistribution(double alpha, double beta)
        throws NotStrictlyPositiveException {
        this(alpha, beta, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public WeibullDistribution(double alpha, double beta,
                               double inverseCumAccuracy) {
        this(new Well19937c(), alpha, beta, inverseCumAccuracy);
    }

    
    public WeibullDistribution(RandomGenerator rng, double alpha, double beta)
        throws NotStrictlyPositiveException {
        this(rng, alpha, beta, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public WeibullDistribution(RandomGenerator rng,
                               double alpha,
                               double beta,
                               double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        super(rng);

        if (alpha <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SHAPE,
                                                   alpha);
        }
        if (beta <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SCALE,
                                                   beta);
        }
        scale = beta;
        shape = alpha;
        solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    
    public double getShape() {
        return shape;
    }

    
    public double getScale() {
        return scale;
    }

    
    public double density(double x) {
        if (x < 0) {
            return 0;
        }

        final double xscale = x / scale;
        final double xscalepow = FastMath.pow(xscale, shape - 1);

        /*
         * FastMath.pow(x / scale, shape) =
         * FastMath.pow(xscale, shape) =
         * FastMath.pow(xscale, shape - 1) * xscale
         */
        final double xscalepowshape = xscalepow * xscale;

        return (shape / scale) * xscalepow * FastMath.exp(-xscalepowshape);
    }

    
    @Override
    public double logDensity(double x) {
        if (x < 0) {
            return Double.NEGATIVE_INFINITY;
        }

        final double xscale = x / scale;
        final double logxscalepow = FastMath.log(xscale) * (shape - 1);

        /*
         * FastMath.pow(x / scale, shape) =
         * FastMath.pow(xscale, shape) =
         * FastMath.pow(xscale, shape - 1) * xscale
         */
        final double xscalepowshape = FastMath.exp(logxscalepow) * xscale;

        return FastMath.log(shape / scale) + logxscalepow - xscalepowshape;
    }

    
    public double cumulativeProbability(double x) {
        double ret;
        if (x <= 0.0) {
            ret = 0.0;
        } else {
            ret = 1.0 - FastMath.exp(-FastMath.pow(x / scale, shape));
        }
        return ret;
    }

    
    @Override
    public double inverseCumulativeProbability(double p) {
        double ret;
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0.0, 1.0);
        } else if (p == 0) {
            ret = 0.0;
        } else  if (p == 1) {
            ret = Double.POSITIVE_INFINITY;
        } else {
            ret = scale * FastMath.pow(-FastMath.log1p(-p), 1.0 / shape);
        }
        return ret;
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        if (!numericalMeanIsCalculated) {
            numericalMean = calculateNumericalMean();
            numericalMeanIsCalculated = true;
        }
        return numericalMean;
    }

    
    protected double calculateNumericalMean() {
        final double sh = getShape();
        final double sc = getScale();

        return sc * FastMath.exp(Gamma.logGamma(1 + (1 / sh)));
    }

    
    public double getNumericalVariance() {
        if (!numericalVarianceIsCalculated) {
            numericalVariance = calculateNumericalVariance();
            numericalVarianceIsCalculated = true;
        }
        return numericalVariance;
    }

    
    protected double calculateNumericalVariance() {
        final double sh = getShape();
        final double sc = getScale();
        final double mn = getNumericalMean();

        return (sc * sc) * FastMath.exp(Gamma.logGamma(1 + (2 / sh))) -
               (mn * mn);
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
}

