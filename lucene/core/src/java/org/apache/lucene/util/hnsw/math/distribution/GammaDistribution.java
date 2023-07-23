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
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.special.Gamma;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class GammaDistribution extends AbstractRealDistribution {
    
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    
    private static final long serialVersionUID = 20120524L;
    
    private final double shape;
    
    private final double scale;
    
    private final double shiftedShape;
    
    private final double densityPrefactor1;
    
    private final double logDensityPrefactor1;
    
    private final double densityPrefactor2;
    
    private final double logDensityPrefactor2;
    
    private final double minY;
    
    private final double maxLogY;
    
    private final double solverAbsoluteAccuracy;

    
    public GammaDistribution(double shape, double scale) throws NotStrictlyPositiveException {
        this(shape, scale, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public GammaDistribution(double shape, double scale, double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        this(new Well19937c(), shape, scale, inverseCumAccuracy);
    }

    
    public GammaDistribution(RandomGenerator rng, double shape, double scale)
        throws NotStrictlyPositiveException {
        this(rng, shape, scale, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    
    public GammaDistribution(RandomGenerator rng,
                             double shape,
                             double scale,
                             double inverseCumAccuracy)
        throws NotStrictlyPositiveException {
        super(rng);

        if (shape <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SHAPE, shape);
        }
        if (scale <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SCALE, scale);
        }

        this.shape = shape;
        this.scale = scale;
        this.solverAbsoluteAccuracy = inverseCumAccuracy;
        this.shiftedShape = shape + Gamma.LANCZOS_G + 0.5;
        final double aux = FastMath.E / (2.0 * FastMath.PI * shiftedShape);
        this.densityPrefactor2 = shape * FastMath.sqrt(aux) / Gamma.lanczos(shape);
        this.logDensityPrefactor2 = FastMath.log(shape) + 0.5 * FastMath.log(aux) -
                                    FastMath.log(Gamma.lanczos(shape));
        this.densityPrefactor1 = this.densityPrefactor2 / scale *
                FastMath.pow(shiftedShape, -shape) *
                FastMath.exp(shape + Gamma.LANCZOS_G);
        this.logDensityPrefactor1 = this.logDensityPrefactor2 - FastMath.log(scale) -
                FastMath.log(shiftedShape) * shape +
                shape + Gamma.LANCZOS_G;
        this.minY = shape + Gamma.LANCZOS_G - FastMath.log(Double.MAX_VALUE);
        this.maxLogY = FastMath.log(Double.MAX_VALUE) / (shape - 1.0);
    }

    
    @Deprecated
    public double getAlpha() {
        return shape;
    }

    
    public double getShape() {
        return shape;
    }

    
    @Deprecated
    public double getBeta() {
        return scale;
    }

    
    public double getScale() {
        return scale;
    }

    
    public double density(double x) {
       /* The present method must return the value of
        *
        *     1       x a     - x
        * ---------- (-)  exp(---)
        * x Gamma(a)  b        b
        *
        * where a is the shape parameter, and b the scale parameter.
        * Substituting the Lanczos approximation of Gamma(a) leads to the
        * following expression of the density
        *
        * a              e            1         y      a
        * - sqrt(------------------) ---- (-----------)  exp(a - y + g),
        * x      2 pi (a + g + 0.5)  L(a)  a + g + 0.5
        *
        * where y = x / b. The above formula is the "natural" computation, which
        * is implemented when no overflow is likely to occur. If overflow occurs
        * with the natural computation, the following identity is used. It is
        * based on the BOOST library
        * http://www.boost.org/doc/libs/1_35_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_gamma/igamma.html
        * Formula (15) needs adaptations, which are detailed below.
        *
        *       y      a
        * (-----------)  exp(a - y + g)
        *  a + g + 0.5
        *                              y - a - g - 0.5    y (g + 0.5)
        *               = exp(a log1pm(---------------) - ----------- + g),
        *                                a + g + 0.5      a + g + 0.5
        *
        *  where log1pm(z) = log(1 + z) - z. Therefore, the value to be
        *  returned is
        *
        * a              e            1
        * - sqrt(------------------) ----
        * x      2 pi (a + g + 0.5)  L(a)
        *                              y - a - g - 0.5    y (g + 0.5)
        *               * exp(a log1pm(---------------) - ----------- + g).
        *                                a + g + 0.5      a + g + 0.5
        */
        if (x < 0) {
            return 0;
        }
        final double y = x / scale;
        if ((y <= minY) || (FastMath.log(y) >= maxLogY)) {
            /*
             * Overflow.
             */
            final double aux1 = (y - shiftedShape) / shiftedShape;
            final double aux2 = shape * (FastMath.log1p(aux1) - aux1);
            final double aux3 = -y * (Gamma.LANCZOS_G + 0.5) / shiftedShape +
                    Gamma.LANCZOS_G + aux2;
            return densityPrefactor2 / x * FastMath.exp(aux3);
        }
        /*
         * Natural calculation.
         */
        return densityPrefactor1 * FastMath.exp(-y) * FastMath.pow(y, shape - 1);
    }

    
    @Override
    public double logDensity(double x) {
        /*
         * see the comment in {@link #density(double)} for computation details
         */
        if (x < 0) {
            return Double.NEGATIVE_INFINITY;
        }
        final double y = x / scale;
        if ((y <= minY) || (FastMath.log(y) >= maxLogY)) {
            /*
             * Overflow.
             */
            final double aux1 = (y - shiftedShape) / shiftedShape;
            final double aux2 = shape * (FastMath.log1p(aux1) - aux1);
            final double aux3 = -y * (Gamma.LANCZOS_G + 0.5) / shiftedShape +
                    Gamma.LANCZOS_G + aux2;
            return logDensityPrefactor2 - FastMath.log(x) + aux3;
        }
        /*
         * Natural calculation.
         */
        return logDensityPrefactor1 - y + FastMath.log(y) * (shape - 1);
    }

    
    public double cumulativeProbability(double x) {
        double ret;

        if (x <= 0) {
            ret = 0;
        } else {
            ret = Gamma.regularizedGammaP(shape, x / scale);
        }

        return ret;
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double getNumericalMean() {
        return shape * scale;
    }

    
    public double getNumericalVariance() {
        return shape * scale * scale;
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
        if (shape < 1) {
            // [1]: p. 228, Algorithm GS

            while (true) {
                // Step 1:
                final double u = random.nextDouble();
                final double bGS = 1 + shape / FastMath.E;
                final double p = bGS * u;

                if (p <= 1) {
                    // Step 2:

                    final double x = FastMath.pow(p, 1 / shape);
                    final double u2 = random.nextDouble();

                    if (u2 > FastMath.exp(-x)) {
                        // Reject
                        continue;
                    } else {
                        return scale * x;
                    }
                } else {
                    // Step 3:

                    final double x = -1 * FastMath.log((bGS - p) / shape);
                    final double u2 = random.nextDouble();

                    if (u2 > FastMath.pow(x, shape - 1)) {
                        // Reject
                        continue;
                    } else {
                        return scale * x;
                    }
                }
            }
        }

        // Now shape >= 1

        final double d = shape - 0.333333333333333333;
        final double c = 1 / (3 * FastMath.sqrt(d));

        while (true) {
            final double x = random.nextGaussian();
            final double v = (1 + c * x) * (1 + c * x) * (1 + c * x);

            if (v <= 0) {
                continue;
            }

            final double x2 = x * x;
            final double u = random.nextDouble();

            // Squeeze
            if (u < 1 - 0.0331 * x2 * x2) {
                return scale * d * v;
            }

            if (FastMath.log(u) < 0.5 * x2 + d * (1 - v + FastMath.log(v))) {
                return scale * d * v;
            }
        }
    }
}
