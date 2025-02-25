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

package org.apache.lucene.util.hnsw.math.linear;

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class QRDecomposition {
    
    private double[][] qrt;
    
    private double[] rDiag;
    
    private RealMatrix cachedQ;
    
    private RealMatrix cachedQT;
    
    private RealMatrix cachedR;
    
    private RealMatrix cachedH;
    
    private final double threshold;

    
    public QRDecomposition(RealMatrix matrix) {
        this(matrix, 0d);
    }

    
    public QRDecomposition(RealMatrix matrix,
                           double threshold) {
        this.threshold = threshold;

        final int m = matrix.getRowDimension();
        final int n = matrix.getColumnDimension();
        qrt = matrix.transpose().getData();
        rDiag = new double[FastMath.min(m, n)];
        cachedQ  = null;
        cachedQT = null;
        cachedR  = null;
        cachedH  = null;

        decompose(qrt);

    }

    
    protected void decompose(double[][] matrix) {
        for (int minor = 0; minor < FastMath.min(matrix.length, matrix[0].length); minor++) {
            performHouseholderReflection(minor, matrix);
        }
    }

    
    protected void performHouseholderReflection(int minor, double[][] matrix) {

        final double[] qrtMinor = matrix[minor];

        /*
         * Let x be the first column of the minor, and a^2 = |x|^2.
         * x will be in the positions qr[minor][minor] through qr[m][minor].
         * The first column of the transformed minor will be (a,0,0,..)'
         * The sign of a is chosen to be opposite to the sign of the first
         * component of x. Let's find a:
         */
        double xNormSqr = 0;
        for (int row = minor; row < qrtMinor.length; row++) {
            final double c = qrtMinor[row];
            xNormSqr += c * c;
        }
        final double a = (qrtMinor[minor] > 0) ? -FastMath.sqrt(xNormSqr) : FastMath.sqrt(xNormSqr);
        rDiag[minor] = a;

        if (a != 0.0) {

            /*
             * Calculate the normalized reflection vector v and transform
             * the first column. We know the norm of v beforehand: v = x-ae
             * so |v|^2 = <x-ae,x-ae> = <x,x>-2a<x,e>+a^2<e,e> =
             * a^2+a^2-2a<x,e> = 2a*(a - <x,e>).
             * Here <x, e> is now qr[minor][minor].
             * v = x-ae is stored in the column at qr:
             */
            qrtMinor[minor] -= a; // now |v|^2 = -2a*(qr[minor][minor])

            /*
             * Transform the rest of the columns of the minor:
             * They will be transformed by the matrix H = I-2vv'/|v|^2.
             * If x is a column vector of the minor, then
             * Hx = (I-2vv'/|v|^2)x = x-2vv'x/|v|^2 = x - 2<x,v>/|v|^2 v.
             * Therefore the transformation is easily calculated by
             * subtracting the column vector (2<x,v>/|v|^2)v from x.
             *
             * Let 2<x,v>/|v|^2 = alpha. From above we have
             * |v|^2 = -2a*(qr[minor][minor]), so
             * alpha = -<x,v>/(a*qr[minor][minor])
             */
            for (int col = minor+1; col < matrix.length; col++) {
                final double[] qrtCol = matrix[col];
                double alpha = 0;
                for (int row = minor; row < qrtCol.length; row++) {
                    alpha -= qrtCol[row] * qrtMinor[row];
                }
                alpha /= a * qrtMinor[minor];

                // Subtract the column vector alpha*v from x.
                for (int row = minor; row < qrtCol.length; row++) {
                    qrtCol[row] -= alpha * qrtMinor[row];
                }
            }
        }
    }


    
    public RealMatrix getR() {

        if (cachedR == null) {

            // R is supposed to be m x n
            final int n = qrt.length;
            final int m = qrt[0].length;
            double[][] ra = new double[m][n];
            // copy the diagonal from rDiag and the upper triangle of qr
            for (int row = FastMath.min(m, n) - 1; row >= 0; row--) {
                ra[row][row] = rDiag[row];
                for (int col = row + 1; col < n; col++) {
                    ra[row][col] = qrt[col][row];
                }
            }
            cachedR = MatrixUtils.createRealMatrix(ra);
        }

        // return the cached matrix
        return cachedR;
    }

    
    public RealMatrix getQ() {
        if (cachedQ == null) {
            cachedQ = getQT().transpose();
        }
        return cachedQ;
    }

    
    public RealMatrix getQT() {
        if (cachedQT == null) {

            // QT is supposed to be m x m
            final int n = qrt.length;
            final int m = qrt[0].length;
            double[][] qta = new double[m][m];

            /*
             * Q = Q1 Q2 ... Q_m, so Q is formed by first constructing Q_m and then
             * applying the Householder transformations Q_(m-1),Q_(m-2),...,Q1 in
             * succession to the result
             */
            for (int minor = m - 1; minor >= FastMath.min(m, n); minor--) {
                qta[minor][minor] = 1.0d;
            }

            for (int minor = FastMath.min(m, n)-1; minor >= 0; minor--){
                final double[] qrtMinor = qrt[minor];
                qta[minor][minor] = 1.0d;
                if (qrtMinor[minor] != 0.0) {
                    for (int col = minor; col < m; col++) {
                        double alpha = 0;
                        for (int row = minor; row < m; row++) {
                            alpha -= qta[col][row] * qrtMinor[row];
                        }
                        alpha /= rDiag[minor] * qrtMinor[minor];

                        for (int row = minor; row < m; row++) {
                            qta[col][row] += -alpha * qrtMinor[row];
                        }
                    }
                }
            }
            cachedQT = MatrixUtils.createRealMatrix(qta);
        }

        // return the cached matrix
        return cachedQT;
    }

    
    public RealMatrix getH() {
        if (cachedH == null) {

            final int n = qrt.length;
            final int m = qrt[0].length;
            double[][] ha = new double[m][n];
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < FastMath.min(i + 1, n); ++j) {
                    ha[i][j] = qrt[j][i] / -rDiag[j];
                }
            }
            cachedH = MatrixUtils.createRealMatrix(ha);
        }

        // return the cached matrix
        return cachedH;
    }

    
    public DecompositionSolver getSolver() {
        return new Solver(qrt, rDiag, threshold);
    }

    
    private static class Solver implements DecompositionSolver {
        
        private final double[][] qrt;
        
        private final double[] rDiag;
        
        private final double threshold;

        
        private Solver(final double[][] qrt,
                       final double[] rDiag,
                       final double threshold) {
            this.qrt   = qrt;
            this.rDiag = rDiag;
            this.threshold = threshold;
        }

        
        public boolean isNonSingular() {
            for (double diag : rDiag) {
                if (FastMath.abs(diag) <= threshold) {
                    return false;
                }
            }
            return true;
        }

        
        public RealVector solve(RealVector b) {
            final int n = qrt.length;
            final int m = qrt[0].length;
            if (b.getDimension() != m) {
                throw new DimensionMismatchException(b.getDimension(), m);
            }
            if (!isNonSingular()) {
                throw new SingularMatrixException();
            }

            final double[] x = new double[n];
            final double[] y = b.toArray();

            // apply Householder transforms to solve Q.y = b
            for (int minor = 0; minor < FastMath.min(m, n); minor++) {

                final double[] qrtMinor = qrt[minor];
                double dotProduct = 0;
                for (int row = minor; row < m; row++) {
                    dotProduct += y[row] * qrtMinor[row];
                }
                dotProduct /= rDiag[minor] * qrtMinor[minor];

                for (int row = minor; row < m; row++) {
                    y[row] += dotProduct * qrtMinor[row];
                }
            }

            // solve triangular system R.x = y
            for (int row = rDiag.length - 1; row >= 0; --row) {
                y[row] /= rDiag[row];
                final double yRow = y[row];
                final double[] qrtRow = qrt[row];
                x[row] = yRow;
                for (int i = 0; i < row; i++) {
                    y[i] -= yRow * qrtRow[i];
                }
            }

            return new ArrayRealVector(x, false);
        }

        
        public RealMatrix solve(RealMatrix b) {
            final int n = qrt.length;
            final int m = qrt[0].length;
            if (b.getRowDimension() != m) {
                throw new DimensionMismatchException(b.getRowDimension(), m);
            }
            if (!isNonSingular()) {
                throw new SingularMatrixException();
            }

            final int columns        = b.getColumnDimension();
            final int blockSize      = BlockRealMatrix.BLOCK_SIZE;
            final int cBlocks        = (columns + blockSize - 1) / blockSize;
            final double[][] xBlocks = BlockRealMatrix.createBlocksLayout(n, columns);
            final double[][] y       = new double[b.getRowDimension()][blockSize];
            final double[]   alpha   = new double[blockSize];

            for (int kBlock = 0; kBlock < cBlocks; ++kBlock) {
                final int kStart = kBlock * blockSize;
                final int kEnd   = FastMath.min(kStart + blockSize, columns);
                final int kWidth = kEnd - kStart;

                // get the right hand side vector
                b.copySubMatrix(0, m - 1, kStart, kEnd - 1, y);

                // apply Householder transforms to solve Q.y = b
                for (int minor = 0; minor < FastMath.min(m, n); minor++) {
                    final double[] qrtMinor = qrt[minor];
                    final double factor     = 1.0 / (rDiag[minor] * qrtMinor[minor]);

                    Arrays.fill(alpha, 0, kWidth, 0.0);
                    for (int row = minor; row < m; ++row) {
                        final double   d    = qrtMinor[row];
                        final double[] yRow = y[row];
                        for (int k = 0; k < kWidth; ++k) {
                            alpha[k] += d * yRow[k];
                        }
                    }
                    for (int k = 0; k < kWidth; ++k) {
                        alpha[k] *= factor;
                    }

                    for (int row = minor; row < m; ++row) {
                        final double   d    = qrtMinor[row];
                        final double[] yRow = y[row];
                        for (int k = 0; k < kWidth; ++k) {
                            yRow[k] += alpha[k] * d;
                        }
                    }
                }

                // solve triangular system R.x = y
                for (int j = rDiag.length - 1; j >= 0; --j) {
                    final int      jBlock = j / blockSize;
                    final int      jStart = jBlock * blockSize;
                    final double   factor = 1.0 / rDiag[j];
                    final double[] yJ     = y[j];
                    final double[] xBlock = xBlocks[jBlock * cBlocks + kBlock];
                    int index = (j - jStart) * kWidth;
                    for (int k = 0; k < kWidth; ++k) {
                        yJ[k]          *= factor;
                        xBlock[index++] = yJ[k];
                    }

                    final double[] qrtJ = qrt[j];
                    for (int i = 0; i < j; ++i) {
                        final double rIJ  = qrtJ[i];
                        final double[] yI = y[i];
                        for (int k = 0; k < kWidth; ++k) {
                            yI[k] -= yJ[k] * rIJ;
                        }
                    }
                }
            }

            return new BlockRealMatrix(n, columns, xBlocks, false);
        }

        
        public RealMatrix getInverse() {
            return solve(MatrixUtils.createRealIdentityMatrix(qrt[0].length));
        }
    }
}
