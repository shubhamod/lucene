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

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.analysis.FunctionUtils;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.DifferentiableUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.ParametricUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class Sigmoid implements UnivariateDifferentiableFunction, DifferentiableUnivariateFunction {
    
    private final double lo;
    
    private final double hi;

    
    public Sigmoid() {
        this(0, 1);
    }

    
    public Sigmoid(double lo,
                   double hi) {
        this.lo = lo;
        this.hi = hi;
    }

    
    @Deprecated
    public UnivariateFunction derivative() {
        return FunctionUtils.toDifferentiableUnivariateFunction(this).derivative();
    }

    
    public double value(double x) {
        return value(x, lo, hi);
    }

    
    public static class Parametric implements ParametricUnivariateFunction {
        
        public double value(double x, double ... param)
            throws NullArgumentException,
                   DimensionMismatchException {
            validateParameters(param);
            return Sigmoid.value(x, param[0], param[1]);
        }

        
        public double[] gradient(double x, double ... param)
            throws NullArgumentException,
                   DimensionMismatchException {
            validateParameters(param);

            final double invExp1 = 1 / (1 + FastMath.exp(-x));

            return new double[] { 1 - invExp1, invExp1 };
        }

        
        private void validateParameters(double[] param)
            throws NullArgumentException,
                   DimensionMismatchException {
            if (param == null) {
                throw new NullArgumentException();
            }
            if (param.length != 2) {
                throw new DimensionMismatchException(param.length, 2);
            }
        }
    }

    
    private static double value(double x,
                                double lo,
                                double hi) {
        return lo + (hi - lo) / (1 + FastMath.exp(-x));
    }

    
    public DerivativeStructure value(final DerivativeStructure t)
        throws DimensionMismatchException {

        double[] f = new double[t.getOrder() + 1];
        final double exp = FastMath.exp(-t.getValue());
        if (Double.isInfinite(exp)) {

            // special handling near lower boundary, to avoid NaN
            f[0] = lo;
            Arrays.fill(f, 1, f.length, 0.0);

        } else {

            // the nth order derivative of sigmoid has the form:
            // dn(sigmoid(x)/dxn = P_n(exp(-x)) / (1+exp(-x))^(n+1)
            // where P_n(t) is a degree n polynomial with normalized higher term
            // P_0(t) = 1, P_1(t) = t, P_2(t) = t^2 - t, P_3(t) = t^3 - 4 t^2 + t...
            // the general recurrence relation for P_n is:
            // P_n(x) = n t P_(n-1)(t) - t (1 + t) P_(n-1)'(t)
            final double[] p = new double[f.length];

            final double inv   = 1 / (1 + exp);
            double coeff = hi - lo;
            for (int n = 0; n < f.length; ++n) {

                // update and evaluate polynomial P_n(t)
                double v = 0;
                p[n] = 1;
                for (int k = n; k >= 0; --k) {
                    v = v * exp + p[k];
                    if (k > 1) {
                        p[k - 1] = (n - k + 2) * p[k - 2] - (k - 1) * p[k - 1];
                    } else {
                        p[0] = 0;
                    }
                }

                coeff *= inv;
                f[n]   = coeff * v;

            }

            // fix function value
            f[0] += lo;

        }

        return t.compose(f);

    }

}
