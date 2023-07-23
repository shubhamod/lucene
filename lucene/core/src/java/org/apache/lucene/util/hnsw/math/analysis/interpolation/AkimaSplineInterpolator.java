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

import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialFunction;
import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialSplineFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class AkimaSplineInterpolator
    implements UnivariateInterpolator {
    
    private static final int MINIMUM_NUMBER_POINTS = 5;

    
    public PolynomialSplineFunction interpolate(double[] xvals,
                                                double[] yvals)
        throws DimensionMismatchException,
               NumberIsTooSmallException,
               NonMonotonicSequenceException {
        if (xvals == null ||
            yvals == null) {
            throw new NullArgumentException();
        }

        if (xvals.length != yvals.length) {
            throw new DimensionMismatchException(xvals.length, yvals.length);
        }

        if (xvals.length < MINIMUM_NUMBER_POINTS) {
            throw new NumberIsTooSmallException(LocalizedFormats.NUMBER_OF_POINTS,
                                                xvals.length,
                                                MINIMUM_NUMBER_POINTS, true);
        }

        MathArrays.checkOrder(xvals);

        final int numberOfDiffAndWeightElements = xvals.length - 1;

        final double[] differences = new double[numberOfDiffAndWeightElements];
        final double[] weights = new double[numberOfDiffAndWeightElements];

        for (int i = 0; i < differences.length; i++) {
            differences[i] = (yvals[i + 1] - yvals[i]) / (xvals[i + 1] - xvals[i]);
        }

        for (int i = 1; i < weights.length; i++) {
            weights[i] = FastMath.abs(differences[i] - differences[i - 1]);
        }

        // Prepare Hermite interpolation scheme.
        final double[] firstDerivatives = new double[xvals.length];

        for (int i = 2; i < firstDerivatives.length - 2; i++) {
            final double wP = weights[i + 1];
            final double wM = weights[i - 1];
            if (Precision.equals(wP, 0.0) &&
                Precision.equals(wM, 0.0)) {
                final double xv = xvals[i];
                final double xvP = xvals[i + 1];
                final double xvM = xvals[i - 1];
                firstDerivatives[i] = (((xvP - xv) * differences[i - 1]) + ((xv - xvM) * differences[i])) / (xvP - xvM);
            } else {
                firstDerivatives[i] = ((wP * differences[i - 1]) + (wM * differences[i])) / (wP + wM);
            }
        }

        firstDerivatives[0] = differentiateThreePoint(xvals, yvals, 0, 0, 1, 2);
        firstDerivatives[1] = differentiateThreePoint(xvals, yvals, 1, 0, 1, 2);
        firstDerivatives[xvals.length - 2] = differentiateThreePoint(xvals, yvals, xvals.length - 2,
                                                                     xvals.length - 3, xvals.length - 2,
                                                                     xvals.length - 1);
        firstDerivatives[xvals.length - 1] = differentiateThreePoint(xvals, yvals, xvals.length - 1,
                                                                     xvals.length - 3, xvals.length - 2,
                                                                     xvals.length - 1);

        return interpolateHermiteSorted(xvals, yvals, firstDerivatives);
    }

    
    private double differentiateThreePoint(double[] xvals, double[] yvals,
                                           int indexOfDifferentiation,
                                           int indexOfFirstSample,
                                           int indexOfSecondsample,
                                           int indexOfThirdSample) {
        final double x0 = yvals[indexOfFirstSample];
        final double x1 = yvals[indexOfSecondsample];
        final double x2 = yvals[indexOfThirdSample];

        final double t = xvals[indexOfDifferentiation] - xvals[indexOfFirstSample];
        final double t1 = xvals[indexOfSecondsample] - xvals[indexOfFirstSample];
        final double t2 = xvals[indexOfThirdSample] - xvals[indexOfFirstSample];

        final double a = (x2 - x0 - (t2 / t1 * (x1 - x0))) / (t2 * t2 - t1 * t2);
        final double b = (x1 - x0 - a * t1 * t1) / t1;

        return (2 * a * t) + b;
    }

    
    private PolynomialSplineFunction interpolateHermiteSorted(double[] xvals,
                                                              double[] yvals,
                                                              double[] firstDerivatives) {
        if (xvals.length != yvals.length) {
            throw new DimensionMismatchException(xvals.length, yvals.length);
        }

        if (xvals.length != firstDerivatives.length) {
            throw new DimensionMismatchException(xvals.length,
                                                 firstDerivatives.length);
        }

        final int minimumLength = 2;
        if (xvals.length < minimumLength) {
            throw new NumberIsTooSmallException(LocalizedFormats.NUMBER_OF_POINTS,
                                                xvals.length, minimumLength,
                                                true);
        }

        final int size = xvals.length - 1;
        final PolynomialFunction[] polynomials = new PolynomialFunction[size];
        final double[] coefficients = new double[4];

        for (int i = 0; i < polynomials.length; i++) {
            final double w = xvals[i + 1] - xvals[i];
            final double w2 = w * w;

            final double yv = yvals[i];
            final double yvP = yvals[i + 1];

            final double fd = firstDerivatives[i];
            final double fdP = firstDerivatives[i + 1];

            coefficients[0] = yv;
            coefficients[1] = firstDerivatives[i];
            coefficients[2] = (3 * (yvP - yv) / w - 2 * fd - fdP) / w;
            coefficients[3] = (2 * (yv - yvP) / w + fd + fdP) / w2;
            polynomials[i] = new PolynomialFunction(coefficients);
        }

        return new PolynomialSplineFunction(xvals, polynomials);

    }
}
