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
package org.apache.lucene.util.hnsw.math.analysis.interpolation;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.Precision;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.jacobian.GaussNewtonOptimizer;
import org.apache.lucene.util.hnsw.math.fitting.PolynomialFitter;
import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialFunction;
import org.apache.lucene.util.hnsw.math.optim.SimpleVectorValueChecker;


@Deprecated
public class SmoothingPolynomialBicubicSplineInterpolator
    extends BicubicSplineInterpolator {
    
    private final PolynomialFitter xFitter;
    
    private final int xDegree;
    
    private final PolynomialFitter yFitter;
    
    private final int yDegree;

    
    public SmoothingPolynomialBicubicSplineInterpolator() {
        this(3);
    }

    
    public SmoothingPolynomialBicubicSplineInterpolator(int degree)
        throws NotPositiveException {
        this(degree, degree);
    }

    
    public SmoothingPolynomialBicubicSplineInterpolator(int xDegree, int yDegree)
        throws NotPositiveException {
        if (xDegree < 0) {
            throw new NotPositiveException(xDegree);
        }
        if (yDegree < 0) {
            throw new NotPositiveException(yDegree);
        }
        this.xDegree = xDegree;
        this.yDegree = yDegree;

        final double safeFactor = 1e2;
        final SimpleVectorValueChecker checker
            = new SimpleVectorValueChecker(safeFactor * Precision.EPSILON,
                                           safeFactor * Precision.SAFE_MIN);
        xFitter = new PolynomialFitter(new GaussNewtonOptimizer(false, checker));
        yFitter = new PolynomialFitter(new GaussNewtonOptimizer(false, checker));
    }

    
    @Override
    public BicubicSplineInterpolatingFunction interpolate(final double[] xval,
                                                          final double[] yval,
                                                          final double[][] fval)
        throws NoDataException, NullArgumentException,
               DimensionMismatchException, NonMonotonicSequenceException {
        if (xval.length == 0 || yval.length == 0 || fval.length == 0) {
            throw new NoDataException();
        }
        if (xval.length != fval.length) {
            throw new DimensionMismatchException(xval.length, fval.length);
        }

        final int xLen = xval.length;
        final int yLen = yval.length;

        for (int i = 0; i < xLen; i++) {
            if (fval[i].length != yLen) {
                throw new DimensionMismatchException(fval[i].length, yLen);
            }
        }

        MathArrays.checkOrder(xval);
        MathArrays.checkOrder(yval);

        // For each line y[j] (0 <= j < yLen), construct a polynomial, with
        // respect to variable x, fitting array fval[][j]
        final PolynomialFunction[] yPolyX = new PolynomialFunction[yLen];
        for (int j = 0; j < yLen; j++) {
            xFitter.clearObservations();
            for (int i = 0; i < xLen; i++) {
                xFitter.addObservedPoint(1, xval[i], fval[i][j]);
            }

            // Initial guess for the fit is zero for each coefficients (of which
            // there are "xDegree" + 1).
            yPolyX[j] = new PolynomialFunction(xFitter.fit(new double[xDegree + 1]));
        }

        // For every knot (xval[i], yval[j]) of the grid, calculate corrected
        // values fval_1
        final double[][] fval_1 = new double[xLen][yLen];
        for (int j = 0; j < yLen; j++) {
            final PolynomialFunction f = yPolyX[j];
            for (int i = 0; i < xLen; i++) {
                fval_1[i][j] = f.value(xval[i]);
            }
        }

        // For each line x[i] (0 <= i < xLen), construct a polynomial, with
        // respect to variable y, fitting array fval_1[i][]
        final PolynomialFunction[] xPolyY = new PolynomialFunction[xLen];
        for (int i = 0; i < xLen; i++) {
            yFitter.clearObservations();
            for (int j = 0; j < yLen; j++) {
                yFitter.addObservedPoint(1, yval[j], fval_1[i][j]);
            }

            // Initial guess for the fit is zero for each coefficients (of which
            // there are "yDegree" + 1).
            xPolyY[i] = new PolynomialFunction(yFitter.fit(new double[yDegree + 1]));
        }

        // For every knot (xval[i], yval[j]) of the grid, calculate corrected
        // values fval_2
        final double[][] fval_2 = new double[xLen][yLen];
        for (int i = 0; i < xLen; i++) {
            final PolynomialFunction f = xPolyY[i];
            for (int j = 0; j < yLen; j++) {
                fval_2[i][j] = f.value(yval[j]);
            }
        }

        return super.interpolate(xval, yval, fval_2);
    }
}
