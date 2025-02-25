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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class LUDecomposition {
    
    private static final double DEFAULT_TOO_SMALL = 1e-11;
    
    private final double[][] lu;
    
    private final int[] pivot;
    
    private boolean even;
    
    private boolean singular;
    
    private RealMatrix cachedL;
    
    private RealMatrix cachedU;
    
    private RealMatrix cachedP;

    
    public LUDecomposition(RealMatrix matrix) {
        this(matrix, DEFAULT_TOO_SMALL);
    }

    
    public LUDecomposition(RealMatrix matrix, double singularityThreshold) {
        if (!matrix.isSquare()) {
            throw new NonSquareMatrixException(matrix.getRowDimension(),
                                               matrix.getColumnDimension());
        }

        final int m = matrix.getColumnDimension();
        lu = matrix.getData();
        pivot = new int[m];
        cachedL = null;
        cachedU = null;
        cachedP = null;

        // Initialize permutation array and parity
        for (int row = 0; row < m; row++) {
            pivot[row] = row;
        }
        even     = true;
        singular = false;

        // Loop over columns
        for (int col = 0; col < m; col++) {

            // upper
            for (int row = 0; row < col; row++) {
                final double[] luRow = lu[row];
                double sum = luRow[col];
                for (int i = 0; i < row; i++) {
                    sum -= luRow[i] * lu[i][col];
                }
                luRow[col] = sum;
            }

            // lower
            int max = col; // permutation row
            double largest = Double.NEGATIVE_INFINITY;
            for (int row = col; row < m; row++) {
                final double[] luRow = lu[row];
                double sum = luRow[col];
                for (int i = 0; i < col; i++) {
                    sum -= luRow[i] * lu[i][col];
                }
                luRow[col] = sum;

                // maintain best permutation choice
                if (FastMath.abs(sum) > largest) {
                    largest = FastMath.abs(sum);
                    max = row;
                }
            }

            // Singularity check
            if (FastMath.abs(lu[max][col]) < singularityThreshold) {
                singular = true;
                return;
            }

            // Pivot if necessary
            if (max != col) {
                double tmp = 0;
                final double[] luMax = lu[max];
                final double[] luCol = lu[col];
                for (int i = 0; i < m; i++) {
                    tmp = luMax[i];
                    luMax[i] = luCol[i];
                    luCol[i] = tmp;
                }
                int temp = pivot[max];
                pivot[max] = pivot[col];
                pivot[col] = temp;
                even = !even;
            }

            // Divide the lower elements by the "winning" diagonal elt.
            final double luDiag = lu[col][col];
            for (int row = col + 1; row < m; row++) {
                lu[row][col] /= luDiag;
            }
        }
    }

    
    public RealMatrix getL() {
        if ((cachedL == null) && !singular) {
            final int m = pivot.length;
            cachedL = MatrixUtils.createRealMatrix(m, m);
            for (int i = 0; i < m; ++i) {
                final double[] luI = lu[i];
                for (int j = 0; j < i; ++j) {
                    cachedL.setEntry(i, j, luI[j]);
                }
                cachedL.setEntry(i, i, 1.0);
            }
        }
        return cachedL;
    }

    
    public RealMatrix getU() {
        if ((cachedU == null) && !singular) {
            final int m = pivot.length;
            cachedU = MatrixUtils.createRealMatrix(m, m);
            for (int i = 0; i < m; ++i) {
                final double[] luI = lu[i];
                for (int j = i; j < m; ++j) {
                    cachedU.setEntry(i, j, luI[j]);
                }
            }
        }
        return cachedU;
    }

    
    public RealMatrix getP() {
        if ((cachedP == null) && !singular) {
            final int m = pivot.length;
            cachedP = MatrixUtils.createRealMatrix(m, m);
            for (int i = 0; i < m; ++i) {
                cachedP.setEntry(i, pivot[i], 1.0);
            }
        }
        return cachedP;
    }

    
    public int[] getPivot() {
        return pivot.clone();
    }

    
    public double getDeterminant() {
        if (singular) {
            return 0;
        } else {
            final int m = pivot.length;
            double determinant = even ? 1 : -1;
            for (int i = 0; i < m; i++) {
                determinant *= lu[i][i];
            }
            return determinant;
        }
    }

    
    public DecompositionSolver getSolver() {
        return new Solver(lu, pivot, singular);
    }

    
    private static class Solver implements DecompositionSolver {

        
        private final double[][] lu;

        
        private final int[] pivot;

        
        private final boolean singular;

        
        private Solver(final double[][] lu, final int[] pivot, final boolean singular) {
            this.lu       = lu;
            this.pivot    = pivot;
            this.singular = singular;
        }

        
        public boolean isNonSingular() {
            return !singular;
        }

        
        public RealVector solve(RealVector b) {
            final int m = pivot.length;
            if (b.getDimension() != m) {
                throw new DimensionMismatchException(b.getDimension(), m);
            }
            if (singular) {
                throw new SingularMatrixException();
            }

            final double[] bp = new double[m];

            // Apply permutations to b
            for (int row = 0; row < m; row++) {
                bp[row] = b.getEntry(pivot[row]);
            }

            // Solve LY = b
            for (int col = 0; col < m; col++) {
                final double bpCol = bp[col];
                for (int i = col + 1; i < m; i++) {
                    bp[i] -= bpCol * lu[i][col];
                }
            }

            // Solve UX = Y
            for (int col = m - 1; col >= 0; col--) {
                bp[col] /= lu[col][col];
                final double bpCol = bp[col];
                for (int i = 0; i < col; i++) {
                    bp[i] -= bpCol * lu[i][col];
                }
            }

            return new ArrayRealVector(bp, false);
        }

        
        public RealMatrix solve(RealMatrix b) {

            final int m = pivot.length;
            if (b.getRowDimension() != m) {
                throw new DimensionMismatchException(b.getRowDimension(), m);
            }
            if (singular) {
                throw new SingularMatrixException();
            }

            final int nColB = b.getColumnDimension();

            // Apply permutations to b
            final double[][] bp = new double[m][nColB];
            for (int row = 0; row < m; row++) {
                final double[] bpRow = bp[row];
                final int pRow = pivot[row];
                for (int col = 0; col < nColB; col++) {
                    bpRow[col] = b.getEntry(pRow, col);
                }
            }

            // Solve LY = b
            for (int col = 0; col < m; col++) {
                final double[] bpCol = bp[col];
                for (int i = col + 1; i < m; i++) {
                    final double[] bpI = bp[i];
                    final double luICol = lu[i][col];
                    for (int j = 0; j < nColB; j++) {
                        bpI[j] -= bpCol[j] * luICol;
                    }
                }
            }

            // Solve UX = Y
            for (int col = m - 1; col >= 0; col--) {
                final double[] bpCol = bp[col];
                final double luDiag = lu[col][col];
                for (int j = 0; j < nColB; j++) {
                    bpCol[j] /= luDiag;
                }
                for (int i = 0; i < col; i++) {
                    final double[] bpI = bp[i];
                    final double luICol = lu[i][col];
                    for (int j = 0; j < nColB; j++) {
                        bpI[j] -= bpCol[j] * luICol;
                    }
                }
            }

            return new Array2DRowRealMatrix(bp, false);
        }

        
        public RealMatrix getInverse() {
            return solve(MatrixUtils.createRealIdentityMatrix(pivot.length));
        }
    }
}
