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


public class Logistic implements UnivariateDifferentiableFunction, DifferentiableUnivariateFunction {
    
    private final double a;
    
    private final double k;
    
    private final double b;
    
    private final double oneOverN;
    
    private final double q;
    
    private final double m;

    
    public Logistic(double k,
                    double m,
                    double b,
                    double q,
                    double a,
                    double n)
        throws NotStrictlyPositiveException {
        if (n <= 0) {
            throw new NotStrictlyPositiveException(n);
        }

        this.k = k;
        this.m = m;
        this.b = b;
        this.q = q;
        this.a = a;
        oneOverN = 1 / n;
    }

    
    public double value(double x) {
        return value(m - x, k, b, q, a, oneOverN);
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
            return Logistic.value(param[1] - x, param[0],
                                  param[2], param[3],
                                  param[4], 1 / param[5]);
        }

        
        public double[] gradient(double x, double ... param)
            throws NullArgumentException,
                   DimensionMismatchException,
                   NotStrictlyPositiveException {
            validateParameters(param);

            final double b = param[2];
            final double q = param[3];

            final double mMinusX = param[1] - x;
            final double oneOverN = 1 / param[5];
            final double exp = FastMath.exp(b * mMinusX);
            final double qExp = q * exp;
            final double qExp1 = qExp + 1;
            final double factor1 = (param[0] - param[4]) * oneOverN / FastMath.pow(qExp1, oneOverN);
            final double factor2 = -factor1 / qExp1;

            // Components of the gradient.
            final double gk = Logistic.value(mMinusX, 1, b, q, 0, oneOverN);
            final double gm = factor2 * b * qExp;
            final double gb = factor2 * mMinusX * qExp;
            final double gq = factor2 * exp;
            final double ga = Logistic.value(mMinusX, 0, b, q, 1, oneOverN);
            final double gn = factor1 * FastMath.log(qExp1) * oneOverN;

            return new double[] { gk, gm, gb, gq, ga, gn };
        }

        
        private void validateParameters(double[] param)
            throws NullArgumentException,
                   DimensionMismatchException,
                   NotStrictlyPositiveException {
            if (param == null) {
                throw new NullArgumentException();
            }
            if (param.length != 6) {
                throw new DimensionMismatchException(param.length, 6);
            }
            if (param[5] <= 0) {
                throw new NotStrictlyPositiveException(param[5]);
            }
        }
    }

    
    private static double value(double mMinusX,
                                double k,
                                double b,
                                double q,
                                double a,
                                double oneOverN) {
        return a + (k - a) / FastMath.pow(1 + q * FastMath.exp(b * mMinusX), oneOverN);
    }

    
    public DerivativeStructure value(final DerivativeStructure t) {
        return t.negate().add(m).multiply(b).exp().multiply(q).add(1).pow(oneOverN).reciprocal().multiply(k - a).add(a);
    }

}
