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
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class BicubicInterpolatingFunction
    implements BivariateFunction {
    
    private static final int NUM_COEFF = 16;
    
    private static final double[][] AINV = {
        { 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0 },
        { -3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0 },
        { 2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0 },
        { 0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0 },
        { 0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0 },
        { -3,0,3,0,0,0,0,0,-2,0,-1,0,0,0,0,0 },
        { 0,0,0,0,-3,0,3,0,0,0,0,0,-2,0,-1,0 },
        { 9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1 },
        { -6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1 },
        { 2,0,-2,0,0,0,0,0,1,0,1,0,0,0,0,0 },
        { 0,0,0,0,2,0,-2,0,0,0,0,0,1,0,1,0 },
        { -6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1 },
        { 4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1 }
    };

    
    private final double[] xval;
    
    private final double[] yval;
    
    private final BicubicFunction[][] splines;

    
    public BicubicInterpolatingFunction(double[] x,
                                        double[] y,
                                        double[][] f,
                                        double[][] dFdX,
                                        double[][] dFdY,
                                        double[][] d2FdXdY)
        throws DimensionMismatchException,
               NoDataException,
               NonMonotonicSequenceException {
        final int xLen = x.length;
        final int yLen = y.length;

        if (xLen == 0 || yLen == 0 || f.length == 0 || f[0].length == 0) {
            throw new NoDataException();
        }
        if (xLen != f.length) {
            throw new DimensionMismatchException(xLen, f.length);
        }
        if (xLen != dFdX.length) {
            throw new DimensionMismatchException(xLen, dFdX.length);
        }
        if (xLen != dFdY.length) {
            throw new DimensionMismatchException(xLen, dFdY.length);
        }
        if (xLen != d2FdXdY.length) {
            throw new DimensionMismatchException(xLen, d2FdXdY.length);
        }

        MathArrays.checkOrder(x);
        MathArrays.checkOrder(y);

        xval = x.clone();
        yval = y.clone();

        final int lastI = xLen - 1;
        final int lastJ = yLen - 1;
        splines = new BicubicFunction[lastI][lastJ];

        for (int i = 0; i < lastI; i++) {
            if (f[i].length != yLen) {
                throw new DimensionMismatchException(f[i].length, yLen);
            }
            if (dFdX[i].length != yLen) {
                throw new DimensionMismatchException(dFdX[i].length, yLen);
            }
            if (dFdY[i].length != yLen) {
                throw new DimensionMismatchException(dFdY[i].length, yLen);
            }
            if (d2FdXdY[i].length != yLen) {
                throw new DimensionMismatchException(d2FdXdY[i].length, yLen);
            }
            final int ip1 = i + 1;
            final double xR = xval[ip1] - xval[i];
            for (int j = 0; j < lastJ; j++) {
                final int jp1 = j + 1;
                final double yR = yval[jp1] - yval[j];
                final double xRyR = xR * yR;
                final double[] beta = new double[] {
                    f[i][j], f[ip1][j], f[i][jp1], f[ip1][jp1],
                    dFdX[i][j] * xR, dFdX[ip1][j] * xR, dFdX[i][jp1] * xR, dFdX[ip1][jp1] * xR,
                    dFdY[i][j] * yR, dFdY[ip1][j] * yR, dFdY[i][jp1] * yR, dFdY[ip1][jp1] * yR,
                    d2FdXdY[i][j] * xRyR, d2FdXdY[ip1][j] * xRyR, d2FdXdY[i][jp1] * xRyR, d2FdXdY[ip1][jp1] * xRyR
                };

                splines[i][j] = new BicubicFunction(computeSplineCoefficients(beta));
            }
        }
    }

    
    public double value(double x, double y)
        throws OutOfRangeException {
        final int i = searchIndex(x, xval);
        final int j = searchIndex(y, yval);

        final double xN = (x - xval[i]) / (xval[i + 1] - xval[i]);
        final double yN = (y - yval[j]) / (yval[j + 1] - yval[j]);

        return splines[i][j].value(xN, yN);
    }

    
    public boolean isValidPoint(double x, double y) {
        if (x < xval[0] ||
            x > xval[xval.length - 1] ||
            y < yval[0] ||
            y > yval[yval.length - 1]) {
            return false;
        } else {
            return true;
        }
    }

    
    private int searchIndex(double c, double[] val) {
        final int r = Arrays.binarySearch(val, c);

        if (r == -1 ||
            r == -val.length - 1) {
            throw new OutOfRangeException(c, val[0], val[val.length - 1]);
        }

        if (r < 0) {
            // "c" in within an interpolation sub-interval: Return the
            // index of the sample at the lower end of the sub-interval.
            return -r - 2;
        }
        final int last = val.length - 1;
        if (r == last) {
            // "c" is the last sample of the range: Return the index
            // of the sample at the lower end of the last sub-interval.
            return last - 1;
        }

        // "c" is another sample point.
        return r;
    }

    
    private double[] computeSplineCoefficients(double[] beta) {
        final double[] a = new double[NUM_COEFF];

        for (int i = 0; i < NUM_COEFF; i++) {
            double result = 0;
            final double[] row = AINV[i];
            for (int j = 0; j < NUM_COEFF; j++) {
                result += row[j] * beta[j];
            }
            a[i] = result;
        }

        return a;
    }
}


class BicubicFunction implements BivariateFunction {
    
    private static final short N = 4;
    
    private final double[][] a;

    
    BicubicFunction(double[] coeff) {
        a = new double[N][N];
        for (int j = 0; j < N; j++) {
            final double[] aJ = a[j];
            for (int i = 0; i < N; i++) {
                aJ[i] = coeff[i * N + j];
            }
        }
    }

    
    public double value(double x, double y) {
        if (x < 0 || x > 1) {
            throw new OutOfRangeException(x, 0, 1);
        }
        if (y < 0 || y > 1) {
            throw new OutOfRangeException(y, 0, 1);
        }

        final double x2 = x * x;
        final double x3 = x2 * x;
        final double[] pX = {1, x, x2, x3};

        final double y2 = y * y;
        final double y3 = y2 * y;
        final double[] pY = {1, y, y2, y3};

        return apply(pX, pY, a);
    }

    
    private double apply(double[] pX, double[] pY, double[][] coeff) {
        double result = 0;
        for (int i = 0; i < N; i++) {
            final double r = MathArrays.linearCombination(coeff[i], pY);
            result += r * pX[i];
        }

        return result;
    }
}
