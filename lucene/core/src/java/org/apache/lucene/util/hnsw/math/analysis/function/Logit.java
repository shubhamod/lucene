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
import org.apache.lucene.util.hnsw.math.analysis.ParametricUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class Logit implements UnivariateDifferentiableFunction, DifferentiableUnivariateFunction {
    
    private final double lo;
    
    private final double hi;

    
    public Logit() {
        this(0, 1);
    }

    
    public Logit(double lo,
                 double hi) {
        this.lo = lo;
        this.hi = hi;
    }

    
    public double value(double x)
        throws OutOfRangeException {
        return value(x, lo, hi);
    }

    
    @Deprecated
    public UnivariateFunction derivative() {
        return FunctionUtils.toDifferentiableUnivariateFunction(this).derivative();
    }

    
    public static class Parametric implements ParametricUnivariateFunction {
        
        public double value(double x, double ... param)
            throws NullArgumentException,
                   DimensionMismatchException {
            validateParameters(param);
            return Logit.value(x, param[0], param[1]);
        }

        
        public double[] gradient(double x, double ... param)
            throws NullArgumentException,
                   DimensionMismatchException {
            validateParameters(param);

            final double lo = param[0];
            final double hi = param[1];

            return new double[] { 1 / (lo - x), 1 / (hi - x) };
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
                                double hi)
        throws OutOfRangeException {
        if (x < lo || x > hi) {
            throw new OutOfRangeException(x, lo, hi);
        }
        return FastMath.log((x - lo) / (hi - x));
    }

    
    public DerivativeStructure value(final DerivativeStructure t)
        throws OutOfRangeException {
        final double x = t.getValue();
        if (x < lo || x > hi) {
            throw new OutOfRangeException(x, lo, hi);
        }
        double[] f = new double[t.getOrder() + 1];

        // function value
        f[0] = FastMath.log((x - lo) / (hi - x));

        if (Double.isInfinite(f[0])) {

            if (f.length > 1) {
                f[1] = Double.POSITIVE_INFINITY;
            }
            // fill the array with infinities
            // (for x close to lo the signs will flip between -inf and +inf,
            //  for x close to hi the signs will always be +inf)
            // this is probably overkill, since the call to compose at the end
            // of the method will transform most infinities into NaN ...
            for (int i = 2; i < f.length; ++i) {
                f[i] = f[i - 2];
            }

        } else {

            // function derivatives
            final double invL = 1.0 / (x - lo);
            double xL = invL;
            final double invH = 1.0 / (hi - x);
            double xH = invH;
            for (int i = 1; i < f.length; ++i) {
                f[i] = xL + xH;
                xL  *= -i * invL;
                xH  *=  i * invH;
            }
        }

        return t.compose(f);
    }
}
