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


@Deprecated
public class BicubicSplineInterpolatingFunction
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
    
    private final BicubicSplineFunction[][] splines;
    
    private final BivariateFunction[][][] partialDerivatives;

    
    public BicubicSplineInterpolatingFunction(double[] x,
                                              double[] y,
                                              double[][] f,
                                              double[][] dFdX,
                                              double[][] dFdY,
                                              double[][] d2FdXdY)
        throws DimensionMismatchException,
               NoDataException,
               NonMonotonicSequenceException {
        this(x, y, f, dFdX, dFdY, d2FdXdY, false);
    }

    
    public BicubicSplineInterpolatingFunction(double[] x,
                                              double[] y,
                                              double[][] f,
                                              double[][] dFdX,
                                              double[][] dFdY,
                                              double[][] d2FdXdY,
                                              boolean initializeDerivatives)
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
        splines = new BicubicSplineFunction[lastI][lastJ];

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
            for (int j = 0; j < lastJ; j++) {
                final int jp1 = j + 1;
                final double[] beta = new double[] {
                    f[i][j], f[ip1][j], f[i][jp1], f[ip1][jp1],
                    dFdX[i][j], dFdX[ip1][j], dFdX[i][jp1], dFdX[ip1][jp1],
                    dFdY[i][j], dFdY[ip1][j], dFdY[i][jp1], dFdY[ip1][jp1],
                    d2FdXdY[i][j], d2FdXdY[ip1][j], d2FdXdY[i][jp1], d2FdXdY[ip1][jp1]
                };

                splines[i][j] = new BicubicSplineFunction(computeSplineCoefficients(beta),
                                                          initializeDerivatives);
            }
        }

        if (initializeDerivatives) {
            // Compute all partial derivatives.
            partialDerivatives = new BivariateFunction[5][lastI][lastJ];

            for (int i = 0; i < lastI; i++) {
                for (int j = 0; j < lastJ; j++) {
                    final BicubicSplineFunction bcs = splines[i][j];
                    partialDerivatives[0][i][j] = bcs.partialDerivativeX();
                    partialDerivatives[1][i][j] = bcs.partialDerivativeY();
                    partialDerivatives[2][i][j] = bcs.partialDerivativeXX();
                    partialDerivatives[3][i][j] = bcs.partialDerivativeYY();
                    partialDerivatives[4][i][j] = bcs.partialDerivativeXY();
                }
            }
        } else {
            // Partial derivative methods cannot be used.
            partialDerivatives = null;
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

    
    public double partialDerivativeX(double x, double y)
        throws OutOfRangeException {
        return partialDerivative(0, x, y);
    }
    
    public double partialDerivativeY(double x, double y)
        throws OutOfRangeException {
        return partialDerivative(1, x, y);
    }
    
    public double partialDerivativeXX(double x, double y)
        throws OutOfRangeException {
        return partialDerivative(2, x, y);
    }
    
    public double partialDerivativeYY(double x, double y)
        throws OutOfRangeException {
        return partialDerivative(3, x, y);
    }
    
    public double partialDerivativeXY(double x, double y)
        throws OutOfRangeException {
        return partialDerivative(4, x, y);
    }

    
    private double partialDerivative(int which, double x, double y)
        throws OutOfRangeException {
        final int i = searchIndex(x, xval);
        final int j = searchIndex(y, yval);

        final double xN = (x - xval[i]) / (xval[i + 1] - xval[i]);
        final double yN = (y - yval[j]) / (yval[j + 1] - yval[j]);

        return partialDerivatives[which][i][j].value(xN, yN);
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


class BicubicSplineFunction implements BivariateFunction {
    
    private static final short N = 4;
    
    private final double[][] a;
    
    private final BivariateFunction partialDerivativeX;
    
    private final BivariateFunction partialDerivativeY;
    
    private final BivariateFunction partialDerivativeXX;
    
    private final BivariateFunction partialDerivativeYY;
    
    private final BivariateFunction partialDerivativeXY;

    
    BicubicSplineFunction(double[] coeff) {
        this(coeff, false);
    }

    
    BicubicSplineFunction(double[] coeff, boolean initializeDerivatives) {
        a = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = coeff[i * N + j];
            }
        }

        if (initializeDerivatives) {
            // Compute all partial derivatives functions.
            final double[][] aX = new double[N][N];
            final double[][] aY = new double[N][N];
            final double[][] aXX = new double[N][N];
            final double[][] aYY = new double[N][N];
            final double[][] aXY = new double[N][N];

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    final double c = a[i][j];
                    aX[i][j] = i * c;
                    aY[i][j] = j * c;
                    aXX[i][j] = (i - 1) * aX[i][j];
                    aYY[i][j] = (j - 1) * aY[i][j];
                    aXY[i][j] = j * aX[i][j];
                }
            }

            partialDerivativeX = new BivariateFunction() {
                    
                    public double value(double x, double y)  {
                        final double x2 = x * x;
                        final double[] pX = {0, 1, x, x2};

                        final double y2 = y * y;
                        final double y3 = y2 * y;
                        final double[] pY = {1, y, y2, y3};

                        return apply(pX, pY, aX);
                    }
                };
            partialDerivativeY = new BivariateFunction() {
                    
                    public double value(double x, double y)  {
                        final double x2 = x * x;
                        final double x3 = x2 * x;
                        final double[] pX = {1, x, x2, x3};

                        final double y2 = y * y;
                        final double[] pY = {0, 1, y, y2};

                        return apply(pX, pY, aY);
                    }
                };
            partialDerivativeXX = new BivariateFunction() {
                    
                    public double value(double x, double y)  {
                        final double[] pX = {0, 0, 1, x};

                        final double y2 = y * y;
                        final double y3 = y2 * y;
                        final double[] pY = {1, y, y2, y3};

                        return apply(pX, pY, aXX);
                    }
                };
            partialDerivativeYY = new BivariateFunction() {
                    
                    public double value(double x, double y)  {
                        final double x2 = x * x;
                        final double x3 = x2 * x;
                        final double[] pX = {1, x, x2, x3};

                        final double[] pY = {0, 0, 1, y};

                        return apply(pX, pY, aYY);
                    }
                };
            partialDerivativeXY = new BivariateFunction() {
                    
                    public double value(double x, double y)  {
                        final double x2 = x * x;
                        final double[] pX = {0, 1, x, x2};

                        final double y2 = y * y;
                        final double[] pY = {0, 1, y, y2};

                        return apply(pX, pY, aXY);
                    }
                };
        } else {
            partialDerivativeX = null;
            partialDerivativeY = null;
            partialDerivativeXX = null;
            partialDerivativeYY = null;
            partialDerivativeXY = null;
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
            for (int j = 0; j < N; j++) {
                result += coeff[i][j] * pX[i] * pY[j];
            }
        }

        return result;
    }

    
    public BivariateFunction partialDerivativeX() {
        return partialDerivativeX;
    }
    
    public BivariateFunction partialDerivativeY() {
        return partialDerivativeY;
    }
    
    public BivariateFunction partialDerivativeXX() {
        return partialDerivativeXX;
    }
    
    public BivariateFunction partialDerivativeYY() {
        return partialDerivativeYY;
    }
    
    public BivariateFunction partialDerivativeXY() {
        return partialDerivativeXY;
    }
}
