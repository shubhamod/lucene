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
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class BicubicInterpolator
    implements BivariateGridInterpolator {
    
    public BicubicInterpolatingFunction interpolate(final double[] xval,
                                                    final double[] yval,
                                                    final double[][] fval)
        throws NoDataException, DimensionMismatchException,
               NonMonotonicSequenceException, NumberIsTooSmallException {
        if (xval.length == 0 || yval.length == 0 || fval.length == 0) {
            throw new NoDataException();
        }
        if (xval.length != fval.length) {
            throw new DimensionMismatchException(xval.length, fval.length);
        }

        MathArrays.checkOrder(xval);
        MathArrays.checkOrder(yval);

        final int xLen = xval.length;
        final int yLen = yval.length;

        // Approximation to the partial derivatives using finite differences.
        final double[][] dFdX = new double[xLen][yLen];
        final double[][] dFdY = new double[xLen][yLen];
        final double[][] d2FdXdY = new double[xLen][yLen];
        for (int i = 1; i < xLen - 1; i++) {
            final int nI = i + 1;
            final int pI = i - 1;

            final double nX = xval[nI];
            final double pX = xval[pI];

            final double deltaX = nX - pX;

            for (int j = 1; j < yLen - 1; j++) {
                final int nJ = j + 1;
                final int pJ = j - 1;

                final double nY = yval[nJ];
                final double pY = yval[pJ];

                final double deltaY = nY - pY;

                dFdX[i][j] = (fval[nI][j] - fval[pI][j]) / deltaX;
                dFdY[i][j] = (fval[i][nJ] - fval[i][pJ]) / deltaY;

                final double deltaXY = deltaX * deltaY;

                d2FdXdY[i][j] = (fval[nI][nJ] - fval[nI][pJ] - fval[pI][nJ] + fval[pI][pJ]) / deltaXY;
            }
        }

        // Create the interpolating function.
        return new BicubicInterpolatingFunction(xval, yval, fval,
                                                dFdX, dFdY, d2FdXdY) {
            
            @Override
            public boolean isValidPoint(double x, double y) {
                if (x < xval[1] ||
                    x > xval[xval.length - 2] ||
                    y < yval[1] ||
                    y > yval[yval.length - 2]) {
                    return false;
                } else {
                    return true;
                }
            }
        };
    }
}
