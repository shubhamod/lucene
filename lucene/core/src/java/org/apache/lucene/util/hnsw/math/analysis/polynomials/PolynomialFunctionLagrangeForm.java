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

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class PolynomialFunctionLagrangeForm implements UnivariateFunction {
    
    private double coefficients[];
    
    private final double x[];
    
    private final double y[];
    
    private boolean coefficientsComputed;

    
    public PolynomialFunctionLagrangeForm(double x[], double y[])
        throws DimensionMismatchException, NumberIsTooSmallException, NonMonotonicSequenceException {
        this.x = new double[x.length];
        this.y = new double[y.length];
        System.arraycopy(x, 0, this.x, 0, x.length);
        System.arraycopy(y, 0, this.y, 0, y.length);
        coefficientsComputed = false;

        if (!verifyInterpolationArray(x, y, false)) {
            MathArrays.sortInPlace(this.x, this.y);
            // Second check in case some abscissa is duplicated.
            verifyInterpolationArray(this.x, this.y, true);
        }
    }

    
    public double value(double z) {
        return evaluateInternal(x, y, z);
    }

    
    public int degree() {
        return x.length - 1;
    }

    
    public double[] getInterpolatingPoints() {
        double[] out = new double[x.length];
        System.arraycopy(x, 0, out, 0, x.length);
        return out;
    }

    
    public double[] getInterpolatingValues() {
        double[] out = new double[y.length];
        System.arraycopy(y, 0, out, 0, y.length);
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

    
    public static double evaluate(double x[], double y[], double z)
        throws DimensionMismatchException, NumberIsTooSmallException, NonMonotonicSequenceException {
        if (verifyInterpolationArray(x, y, false)) {
            return evaluateInternal(x, y, z);
        }

        // Array is not sorted.
        final double[] xNew = new double[x.length];
        final double[] yNew = new double[y.length];
        System.arraycopy(x, 0, xNew, 0, x.length);
        System.arraycopy(y, 0, yNew, 0, y.length);

        MathArrays.sortInPlace(xNew, yNew);
        // Second check in case some abscissa is duplicated.
        verifyInterpolationArray(xNew, yNew, true);
        return evaluateInternal(xNew, yNew, z);
    }

    
    private static double evaluateInternal(double x[], double y[], double z) {
        int nearest = 0;
        final int n = x.length;
        final double[] c = new double[n];
        final double[] d = new double[n];
        double min_dist = Double.POSITIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            // initialize the difference arrays
            c[i] = y[i];
            d[i] = y[i];
            // find out the abscissa closest to z
            final double dist = FastMath.abs(z - x[i]);
            if (dist < min_dist) {
                nearest = i;
                min_dist = dist;
            }
        }

        // initial approximation to the function value at z
        double value = y[nearest];

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < n-i; j++) {
                final double tc = x[j] - z;
                final double td = x[i+j] - z;
                final double divider = x[j] - x[i+j];
                // update the difference arrays
                final double w = (c[j+1] - d[j]) / divider;
                c[j] = tc * w;
                d[j] = td * w;
            }
            // sum up the difference terms to get the final value
            if (nearest < 0.5*(n-i+1)) {
                value += c[nearest];    // fork down
            } else {
                nearest--;
                value += d[nearest];    // fork up
            }
        }

        return value;
    }

    
    protected void computeCoefficients() {
        final int n = degree() + 1;
        coefficients = new double[n];
        for (int i = 0; i < n; i++) {
            coefficients[i] = 0.0;
        }

        // c[] are the coefficients of P(x) = (x-x[0])(x-x[1])...(x-x[n-1])
        final double[] c = new double[n+1];
        c[0] = 1.0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j > 0; j--) {
                c[j] = c[j-1] - c[j] * x[i];
            }
            c[0] *= -x[i];
            c[i+1] = 1;
        }

        final double[] tc = new double[n];
        for (int i = 0; i < n; i++) {
            // d = (x[i]-x[0])...(x[i]-x[i-1])(x[i]-x[i+1])...(x[i]-x[n-1])
            double d = 1;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    d *= x[i] - x[j];
                }
            }
            final double t = y[i] / d;
            // Lagrange polynomial is the sum of n terms, each of which is a
            // polynomial of degree n-1. tc[] are the coefficients of the i-th
            // numerator Pi(x) = (x-x[0])...(x-x[i-1])(x-x[i+1])...(x-x[n-1]).
            tc[n-1] = c[n];     // actually c[n] = 1
            coefficients[n-1] += t * tc[n-1];
            for (int j = n-2; j >= 0; j--) {
                tc[j] = c[j+1] + tc[j+1] * x[i];
                coefficients[j] += t * tc[j];
            }
        }

        coefficientsComputed = true;
    }

    
    public static boolean verifyInterpolationArray(double x[], double y[], boolean abort)
        throws DimensionMismatchException, NumberIsTooSmallException, NonMonotonicSequenceException {
        if (x.length != y.length) {
            throw new DimensionMismatchException(x.length, y.length);
        }
        if (x.length < 2) {
            throw new NumberIsTooSmallException(LocalizedFormats.WRONG_NUMBER_OF_POINTS, 2, x.length, true);
        }

        return MathArrays.checkOrder(x, MathArrays.OrderDirection.INCREASING, true, abort);
    }
}
