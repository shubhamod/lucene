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

import java.util.Arrays;
import org.apache.lucene.util.hnsw.math.analysis.BivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialSplineFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.InsufficientDataException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class PiecewiseBicubicSplineInterpolatingFunction
    implements BivariateFunction {
    
    private static final int MIN_NUM_POINTS = 5;
    
    private final double[] xval;
    
    private final double[] yval;
    
    private final double[][] fval;

    
    public PiecewiseBicubicSplineInterpolatingFunction(double[] x,
                                                       double[] y,
                                                       double[][] f)
        throws DimensionMismatchException,
               NullArgumentException,
               NoDataException,
               NonMonotonicSequenceException {
        if (x == null ||
            y == null ||
            f == null ||
            f[0] == null) {
            throw new NullArgumentException();
        }

        final int xLen = x.length;
        final int yLen = y.length;

        if (xLen == 0 ||
            yLen == 0 ||
            f.length == 0 ||
            f[0].length == 0) {
            throw new NoDataException();
        }

        if (xLen < MIN_NUM_POINTS ||
            yLen < MIN_NUM_POINTS ||
            f.length < MIN_NUM_POINTS ||
            f[0].length < MIN_NUM_POINTS) {
            throw new InsufficientDataException();
        }

        if (xLen != f.length) {
            throw new DimensionMismatchException(xLen, f.length);
        }

        if (yLen != f[0].length) {
            throw new DimensionMismatchException(yLen, f[0].length);
        }

        MathArrays.checkOrder(x);
        MathArrays.checkOrder(y);

        xval = x.clone();
        yval = y.clone();
        fval = f.clone();
    }

    
    public double value(double x,
                        double y)
        throws OutOfRangeException {
        final AkimaSplineInterpolator interpolator = new AkimaSplineInterpolator();
        final int offset = 2;
        final int count = offset + 3;
        final int i = searchIndex(x, xval, offset, count);
        final int j = searchIndex(y, yval, offset, count);

        final double xArray[] = new double[count];
        final double yArray[] = new double[count];
        final double zArray[] = new double[count];
        final double interpArray[] = new double[count];

        for (int index = 0; index < count; index++) {
            xArray[index] = xval[i + index];
            yArray[index] = yval[j + index];
        }

        for (int zIndex = 0; zIndex < count; zIndex++) {
            for (int index = 0; index < count; index++) {
                zArray[index] = fval[i + index][j + zIndex];
            }
            final PolynomialSplineFunction spline = interpolator.interpolate(xArray, zArray);
            interpArray[zIndex] = spline.value(x);
        }

        final PolynomialSplineFunction spline = interpolator.interpolate(yArray, interpArray);

        double returnValue = spline.value(y);

        return returnValue;
    }

    
    public boolean isValidPoint(double x,
                                double y) {
        if (x < xval[0] ||
            x > xval[xval.length - 1] ||
            y < yval[0] ||
            y > yval[yval.length - 1]) {
            return false;
        } else {
            return true;
        }
    }

    
    private int searchIndex(double c,
                            double[] val,
                            int offset,
                            int count) {
        int r = Arrays.binarySearch(val, c);

        if (r == -1 || r == -val.length - 1) {
            throw new OutOfRangeException(c, val[0], val[val.length - 1]);
        }

        if (r < 0) {
            // "c" in within an interpolation sub-interval, which returns
            // negative
            // need to remove the negative sign for consistency
            r = -r - offset - 1;
        } else {
            r -= offset;
        }

        if (r < 0) {
            r = 0;
        }

        if ((r + count) >= val.length) {
            // "c" is the last sample of the range: Return the index
            // of the sample at the lower end of the last sub-interval.
            r = val.length - count;
        }

        return r;
    }
}
