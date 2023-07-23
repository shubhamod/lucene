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

package org.apache.lucene.util.hnsw.math.analysis.function;

import org.apache.lucene.util.hnsw.math.analysis.DifferentiableUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.FunctionUtils;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class Sinc implements UnivariateDifferentiableFunction, DifferentiableUnivariateFunction {
    
    private static final double SHORTCUT = 6.0e-3;
    
    private final boolean normalized;

    
    public Sinc() {
        this(false);
    }

    
    public Sinc(boolean normalized) {
        this.normalized = normalized;
    }

    
    public double value(final double x) {
        final double scaledX = normalized ? FastMath.PI * x : x;
        if (FastMath.abs(scaledX) <= SHORTCUT) {
            // use Taylor series
            final double scaledX2 = scaledX * scaledX;
            return ((scaledX2 - 20) * scaledX2 + 120) / 120;
        } else {
            // use definition expression
            return FastMath.sin(scaledX) / scaledX;
        }
    }

    
    @Deprecated
    public UnivariateFunction derivative() {
        return FunctionUtils.toDifferentiableUnivariateFunction(this).derivative();
    }

    
    public DerivativeStructure value(final DerivativeStructure t)
        throws DimensionMismatchException {

        final double scaledX  = (normalized ? FastMath.PI : 1) * t.getValue();
        final double scaledX2 = scaledX * scaledX;

        double[] f = new double[t.getOrder() + 1];

        if (FastMath.abs(scaledX) <= SHORTCUT) {

            for (int i = 0; i < f.length; ++i) {
                final int k = i / 2;
                if ((i & 0x1) == 0) {
                    // even derivation order
                    f[i] = (((k & 0x1) == 0) ? 1 : -1) *
                           (1.0 / (i + 1) - scaledX2 * (1.0 / (2 * i + 6) - scaledX2 / (24 * i + 120)));
                } else {
                    // odd derivation order
                    f[i] = (((k & 0x1) == 0) ? -scaledX : scaledX) *
                           (1.0 / (i + 2) - scaledX2 * (1.0 / (6 * i + 24) - scaledX2 / (120 * i + 720)));
                }
            }

        } else {

            final double inv = 1 / scaledX;
            final double cos = FastMath.cos(scaledX);
            final double sin = FastMath.sin(scaledX);

            f[0] = inv * sin;

            // the nth order derivative of sinc has the form:
            // dn(sinc(x)/dxn = [S_n(x) sin(x) + C_n(x) cos(x)] / x^(n+1)
            // where S_n(x) is an even polynomial with degree n-1 or n (depending on parity)
            // and C_n(x) is an odd polynomial with degree n-1 or n (depending on parity)
            // S_0(x) = 1, S_1(x) = -1, S_2(x) = -x^2 + 2, S_3(x) = 3x^2 - 6...
            // C_0(x) = 0, C_1(x) = x, C_2(x) = -2x, C_3(x) = -x^3 + 6x...
            // the general recurrence relations for S_n and C_n are:
            // S_n(x) = x S_(n-1)'(x) - n S_(n-1)(x) - x C_(n-1)(x)
            // C_n(x) = x C_(n-1)'(x) - n C_(n-1)(x) + x S_(n-1)(x)
            // as per polynomials parity, we can store both S_n and C_n in the same array
            final double[] sc = new double[f.length];
            sc[0] = 1;

            double coeff = inv;
            for (int n = 1; n < f.length; ++n) {

                double s = 0;
                double c = 0;

                // update and evaluate polynomials S_n(x) and C_n(x)
                final int kStart;
                if ((n & 0x1) == 0) {
                    // even derivation order, S_n is degree n and C_n is degree n-1
                    sc[n] = 0;
                    kStart = n;
                } else {
                    // odd derivation order, S_n is degree n-1 and C_n is degree n
                    sc[n] = sc[n - 1];
                    c = sc[n];
                    kStart = n - 1;
                }

                // in this loop, k is always even
                for (int k = kStart; k > 1; k -= 2) {

                    // sine part
                    sc[k]     = (k - n) * sc[k] - sc[k - 1];
                    s         = s * scaledX2 + sc[k];

                    // cosine part
                    sc[k - 1] = (k - 1 - n) * sc[k - 1] + sc[k -2];
                    c         = c * scaledX2 + sc[k - 1];

                }
                sc[0] *= -n;
                s      = s * scaledX2 + sc[0];

                coeff *= inv;
                f[n]   = coeff * (s * sin + c * scaledX * cos);

            }

        }

        if (normalized) {
            double scale = FastMath.PI;
            for (int i = 1; i < f.length; ++i) {
                f[i]  *= scale;
                scale *= FastMath.PI;
            }
        }

        return t.compose(f);

    }

}
