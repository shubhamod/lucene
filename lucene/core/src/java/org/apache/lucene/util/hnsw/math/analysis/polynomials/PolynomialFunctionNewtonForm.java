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
package org.apache.lucene.util.hnsw.math.analysis.polynomials;

import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class PolynomialFunctionNewtonForm implements UnivariateDifferentiableFunction {

    
    private double coefficients[];

    
    private final double c[];

    
    private final double a[];

    
    private boolean coefficientsComputed;

    
    public PolynomialFunctionNewtonForm(double a[], double c[])
        throws NullArgumentException, NoDataException, DimensionMismatchException {

        verifyInputArray(a, c);
        this.a = new double[a.length];
        this.c = new double[c.length];
        System.arraycopy(a, 0, this.a, 0, a.length);
        System.arraycopy(c, 0, this.c, 0, c.length);
        coefficientsComputed = false;
    }

    
    public double value(double z) {
       return evaluate(a, c, z);
    }

    
    public DerivativeStructure value(final DerivativeStructure t) {
        verifyInputArray(a, c);

        final int n = c.length;
        DerivativeStructure value = new DerivativeStructure(t.getFreeParameters(), t.getOrder(), a[n]);
        for (int i = n - 1; i >= 0; i--) {
            value = t.subtract(c[i]).multiply(value).add(a[i]);
        }

        return value;

    }

    
    public int degree() {
        return c.length;
    }

    
    public double[] getNewtonCoefficients() {
        double[] out = new double[a.length];
        System.arraycopy(a, 0, out, 0, a.length);
        return out;
    }

    
    public double[] getCenters() {
        double[] out = new double[c.length];
        System.arraycopy(c, 0, out, 0, c.length);
        return out;
    }

    
    public double[] getCoefficients() {
        if (!coefficientsComputed) {
            computeCoefficients();
        }
        double[] out = new double[coefficients.length];
        System.arraycopy(coefficients, 0, out, 0, coefficients.length);
        return out;
    }

    
    public static double evaluate(double a[], double c[], double z)
        throws NullArgumentException, DimensionMismatchException, NoDataException {
        verifyInputArray(a, c);

        final int n = c.length;
        double value = a[n];
        for (int i = n - 1; i >= 0; i--) {
            value = a[i] + (z - c[i]) * value;
        }

        return value;
    }

    
    protected void computeCoefficients() {
        final int n = degree();

        coefficients = new double[n+1];
        for (int i = 0; i <= n; i++) {
            coefficients[i] = 0.0;
        }

        coefficients[0] = a[n];
        for (int i = n-1; i >= 0; i--) {
            for (int j = n-i; j > 0; j--) {
                coefficients[j] = coefficients[j-1] - c[i] * coefficients[j];
            }
            coefficients[0] = a[i] - c[i] * coefficients[0];
        }

        coefficientsComputed = true;
    }

    
    protected static void verifyInputArray(double a[], double c[])
        throws NullArgumentException, NoDataException, DimensionMismatchException {
        MathUtils.checkNotNull(a);
        MathUtils.checkNotNull(c);
        if (a.length == 0 || c.length == 0) {
            throw new NoDataException(LocalizedFormats.EMPTY_POLYNOMIALS_COEFFICIENTS_ARRAY);
        }
        if (a.length != c.length + 1) {
            throw new DimensionMismatchException(LocalizedFormats.ARRAY_SIZES_SHOULD_HAVE_DIFFERENCE_1,
                                                 a.length, c.length);
        }
    }

}
