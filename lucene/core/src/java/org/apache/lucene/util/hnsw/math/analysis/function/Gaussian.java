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
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class Gaussian implements UnivariateDifferentiableFunction, DifferentiableUnivariateFunction {
    
    private final double mean;
    
    private final double is;
    
    private final double i2s2;
    
    private final double norm;

    
    public Gaussian(double norm,
                    double mean,
                    double sigma)
        throws NotStrictlyPositiveException {
        if (sigma <= 0) {
            throw new NotStrictlyPositiveException(sigma);
        }

        this.norm = norm;
        this.mean = mean;
        this.is   = 1 / sigma;
        this.i2s2 = 0.5 * is * is;
    }

    
    public Gaussian(double mean,
                    double sigma)
        throws NotStrictlyPositiveException {
        this(1 / (sigma * FastMath.sqrt(2 * Math.PI)), mean, sigma);
    }

    
    public Gaussian() {
        this(0, 1);
    }

    
    public double value(double x) {
        return value(x - mean, norm, i2s2);
    }

    
    @Deprecated
    public UnivariateFunction derivative() {
        return FunctionUtils.toDifferentiableUnivariateFunction(this).derivative();
    }

    
    public static class Parametric implements ParametricUnivariateFunction {
        
        public double value(double x, double ... param)
            throws NullArgumentException,
                   DimensionMismatchException,
                   NotStrictlyPositiveException {
            validateParameters(param);

            final double diff = x - param[1];
            final double i2s2 = 1 / (2 * param[2] * param[2]);
            return Gaussian.value(diff, param[0], i2s2);
        }

        
        public double[] gradient(double x, double ... param)
            throws NullArgumentException,
                   DimensionMismatchException,
                   NotStrictlyPositiveException {
            validateParameters(param);

            final double norm = param[0];
            final double diff = x - param[1];
            final double sigma = param[2];
            final double i2s2 = 1 / (2 * sigma * sigma);

            final double n = Gaussian.value(diff, 1, i2s2);
            final double m = norm * n * 2 * i2s2 * diff;
            final double s = m * diff / sigma;

            return new double[] { n, m, s };
        }

        
        private void validateParameters(double[] param)
            throws NullArgumentException,
                   DimensionMismatchException,
                   NotStrictlyPositiveException {
            if (param == null) {
                throw new NullArgumentException();
            }
            if (param.length != 3) {
                throw new DimensionMismatchException(param.length, 3);
            }
            if (param[2] <= 0) {
                throw new NotStrictlyPositiveException(param[2]);
            }
        }
    }

    
    private static double value(double xMinusMean,
                                double norm,
                                double i2s2) {
        return norm * FastMath.exp(-xMinusMean * xMinusMean * i2s2);
    }

    
    public DerivativeStructure value(final DerivativeStructure t)
        throws DimensionMismatchException {

        final double u = is * (t.getValue() - mean);
        double[] f = new double[t.getOrder() + 1];

        // the nth order derivative of the Gaussian has the form:
        // dn(g(x)/dxn = (norm / s^n) P_n(u) exp(-u^2/2) with u=(x-m)/s
        // where P_n(u) is a degree n polynomial with same parity as n
        // P_0(u) = 1, P_1(u) = -u, P_2(u) = u^2 - 1, P_3(u) = -u^3 + 3 u...
        // the general recurrence relation for P_n is:
        // P_n(u) = P_(n-1)'(u) - u P_(n-1)(u)
        // as per polynomial parity, we can store coefficients of both P_(n-1) and P_n in the same array
        final double[] p = new double[f.length];
        p[0] = 1;
        final double u2 = u * u;
        double coeff = norm * FastMath.exp(-0.5 * u2);
        if (coeff <= Precision.SAFE_MIN) {
            Arrays.fill(f, 0.0);
        } else {
            f[0] = coeff;
            for (int n = 1; n < f.length; ++n) {

                // update and evaluate polynomial P_n(x)
                double v = 0;
                p[n] = -p[n - 1];
                for (int k = n; k >= 0; k -= 2) {
                    v = v * u2 + p[k];
                    if (k > 2) {
                        p[k - 2] = (k - 1) * p[k - 1] - p[k - 3];
                    } else if (k == 2) {
                        p[0] = p[1];
                    }
                }
                if ((n & 0x1) == 1) {
                    v *= u;
                }

                coeff *= is;
                f[n] = coeff * v;

            }
        }

        return t.compose(f);

    }

}
