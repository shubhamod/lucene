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

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.analysis.DifferentiableUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class PolynomialSplineFunction implements UnivariateDifferentiableFunction, DifferentiableUnivariateFunction {
    
    private final double knots[];
    
    private final PolynomialFunction polynomials[];
    
    private final int n;


    
    public PolynomialSplineFunction(double knots[], PolynomialFunction polynomials[])
        throws NullArgumentException, NumberIsTooSmallException,
               DimensionMismatchException, NonMonotonicSequenceException{
        if (knots == null ||
            polynomials == null) {
            throw new NullArgumentException();
        }
        if (knots.length < 2) {
            throw new NumberIsTooSmallException(LocalizedFormats.NOT_ENOUGH_POINTS_IN_SPLINE_PARTITION,
                                                2, knots.length, false);
        }
        if (knots.length - 1 != polynomials.length) {
            throw new DimensionMismatchException(polynomials.length, knots.length);
        }
        MathArrays.checkOrder(knots);

        this.n = knots.length -1;
        this.knots = new double[n + 1];
        System.arraycopy(knots, 0, this.knots, 0, n + 1);
        this.polynomials = new PolynomialFunction[n];
        System.arraycopy(polynomials, 0, this.polynomials, 0, n);
    }

    
    public double value(double v) {
        if (v < knots[0] || v > knots[n]) {
            throw new OutOfRangeException(v, knots[0], knots[n]);
        }
        int i = Arrays.binarySearch(knots, v);
        if (i < 0) {
            i = -i - 2;
        }
        // This will handle the case where v is the last knot value
        // There are only n-1 polynomials, so if v is the last knot
        // then we will use the last polynomial to calculate the value.
        if ( i >= polynomials.length ) {
            i--;
        }
        return polynomials[i].value(v - knots[i]);
    }

    
    public UnivariateFunction derivative() {
        return polynomialSplineDerivative();
    }

    
    public PolynomialSplineFunction polynomialSplineDerivative() {
        PolynomialFunction derivativePolynomials[] = new PolynomialFunction[n];
        for (int i = 0; i < n; i++) {
            derivativePolynomials[i] = polynomials[i].polynomialDerivative();
        }
        return new PolynomialSplineFunction(knots, derivativePolynomials);
    }


    
    public DerivativeStructure value(final DerivativeStructure t) {
        final double t0 = t.getValue();
        if (t0 < knots[0] || t0 > knots[n]) {
            throw new OutOfRangeException(t0, knots[0], knots[n]);
        }
        int i = Arrays.binarySearch(knots, t0);
        if (i < 0) {
            i = -i - 2;
        }
        // This will handle the case where t is the last knot value
        // There are only n-1 polynomials, so if t is the last knot
        // then we will use the last polynomial to calculate the value.
        if ( i >= polynomials.length ) {
            i--;
        }
        return polynomials[i].value(t.subtract(knots[i]));
    }

    
    public int getN() {
        return n;
    }

    
    public PolynomialFunction[] getPolynomials() {
        PolynomialFunction p[] = new PolynomialFunction[n];
        System.arraycopy(polynomials, 0, p, 0, n);
        return p;
    }

    
    public double[] getKnots() {
        double out[] = new double[n + 1];
        System.arraycopy(knots, 0, out, 0, n + 1);
        return out;
    }

    
    public boolean isValidPoint(double x) {
        if (x < knots[0] ||
            x > knots[n]) {
            return false;
        } else {
            return true;
        }
    }
}
