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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class FieldLUDecomposition<T extends FieldElement<T>> {

    
    private final Field<T> field;

    
    private T[][] lu;

    
    private int[] pivot;

    
    private boolean even;

    
    private boolean singular;

    
    private FieldMatrix<T> cachedL;

    
    private FieldMatrix<T> cachedU;

    
    private FieldMatrix<T> cachedP;

    
    public FieldLUDecomposition(FieldMatrix<T> matrix) {
        if (!matrix.isSquare()) {
            throw new NonSquareMatrixException(matrix.getRowDimension(),
                                               matrix.getColumnDimension());
        }

        final int m = matrix.getColumnDimension();
        field = matrix.getField();
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

            T sum = field.getZero();

            // upper
            for (int row = 0; row < col; row++) {
                final T[] luRow = lu[row];
                sum = luRow[col];
                for (int i = 0; i < row; i++) {
                    sum = sum.subtract(luRow[i].multiply(lu[i][col]));
                }
                luRow[col] = sum;
            }

            // lower
            int nonZero = col; // permutation row
            for (int row = col; row < m; row++) {
                final T[] luRow = lu[row];
                sum = luRow[col];
                for (int i = 0; i < col; i++) {
                    sum = sum.subtract(luRow[i].multiply(lu[i][col]));
                }
                luRow[col] = sum;

                if (lu[nonZero][col].equals(field.getZero())) {
                    // try to select a better permutation choice
                    ++nonZero;
                }
            }

            // Singularity check
            if (nonZero >= m) {
                singular = true;
                return;
            }

            // Pivot if necessary
            if (nonZero != col) {
                T tmp = field.getZero();
                for (int i = 0; i < m; i++) {
                    tmp = lu[nonZero][i];
                    lu[nonZero][i] = lu[col][i];
                    lu[col][i] = tmp;
                }
                int temp = pivot[nonZero];
                pivot[nonZero] = pivot[col];
                pivot[col] = temp;
                even = !even;
            }

            // Divide the lower elements by the "winning" diagonal elt.
            final T luDiag = lu[col][col];
            for (int row = col + 1; row < m; row++) {
                final T[] luRow = lu[row];
                luRow[col] = luRow[col].divide(luDiag);
            }
        }

    }

    
    public FieldMatrix<T> getL() {
        if ((cachedL == null) && !singular) {
            final int m = pivot.length;
            cachedL = new Array2DRowFieldMatrix<T>(field, m, m);
            for (int i = 0; i < m; ++i) {
                final T[] luI = lu[i];
                for (int j = 0; j < i; ++j) {
                    cachedL.setEntry(i, j, luI[j]);
                }
                cachedL.setEntry(i, i, field.getOne());
            }
        }
        return cachedL;
    }

    
    public FieldMatrix<T> getU() {
        if ((cachedU == null) && !singular) {
            final int m = pivot.length;
            cachedU = new Array2DRowFieldMatrix<T>(field, m, m);
            for (int i = 0; i < m; ++i) {
                final T[] luI = lu[i];
                for (int j = i; j < m; ++j) {
                    cachedU.setEntry(i, j, luI[j]);
                }
            }
        }
        return cachedU;
    }

    
    public FieldMatrix<T> getP() {
        if ((cachedP == null) && !singular) {
            final int m = pivot.length;
            cachedP = new Array2DRowFieldMatrix<T>(field, m, m);
            for (int i = 0; i < m; ++i) {
                cachedP.setEntry(i, pivot[i], field.getOne());
            }
        }
        return cachedP;
    }

    
    public int[] getPivot() {
        return pivot.clone();
    }

    
    public T getDeterminant() {
        if (singular) {
            return field.getZero();
        } else {
            final int m = pivot.length;
            T determinant = even ? field.getOne() : field.getZero().subtract(field.getOne());
            for (int i = 0; i < m; i++) {
                determinant = determinant.multiply(lu[i][i]);
            }
            return determinant;
        }
    }

    
    public FieldDecompositionSolver<T> getSolver() {
        return new Solver<T>(field, lu, pivot, singular);
    }

    
    private static class Solver<T extends FieldElement<T>> implements FieldDecompositionSolver<T> {

        
        private final Field<T> field;

        
        private final T[][] lu;

        
        private final int[] pivot;

        
        private final boolean singular;

        
        private Solver(final Field<T> field, final T[][] lu,
                       final int[] pivot, final boolean singular) {
            this.field    = field;
            this.lu       = lu;
            this.pivot    = pivot;
            this.singular = singular;
        }

        
        public boolean isNonSingular() {
            return !singular;
        }

        
        public FieldVector<T> solve(FieldVector<T> b) {
            try {
                return solve((ArrayFieldVector<T>) b);
            } catch (ClassCastException cce) {

                final int m = pivot.length;
                if (b.getDimension() != m) {
                    throw new DimensionMismatchException(b.getDimension(), m);
                }
                if (singular) {
                    throw new SingularMatrixException();
                }

                // Apply permutations to b
                final T[] bp = MathArrays.buildArray(field, m);
                for (int row = 0; row < m; row++) {
                    bp[row] = b.getEntry(pivot[row]);
                }

                // Solve LY = b
                for (int col = 0; col < m; col++) {
                    final T bpCol = bp[col];
                    for (int i = col + 1; i < m; i++) {
                        bp[i] = bp[i].subtract(bpCol.multiply(lu[i][col]));
                    }
                }

                // Solve UX = Y
                for (int col = m - 1; col >= 0; col--) {
                    bp[col] = bp[col].divide(lu[col][col]);
                    final T bpCol = bp[col];
                    for (int i = 0; i < col; i++) {
                        bp[i] = bp[i].subtract(bpCol.multiply(lu[i][col]));
                    }
                }

                return new ArrayFieldVector<T>(field, bp, false);

            }
        }

        
        public ArrayFieldVector<T> solve(ArrayFieldVector<T> b) {
            final int m = pivot.length;
            final int length = b.getDimension();
            if (length != m) {
                throw new DimensionMismatchException(length, m);
            }
            if (singular) {
                throw new SingularMatrixException();
            }

            // Apply permutations to b
            final T[] bp = MathArrays.buildArray(field, m);
            for (int row = 0; row < m; row++) {
                bp[row] = b.getEntry(pivot[row]);
            }

            // Solve LY = b
            for (int col = 0; col < m; col++) {
                final T bpCol = bp[col];
                for (int i = col + 1; i < m; i++) {
                    bp[i] = bp[i].subtract(bpCol.multiply(lu[i][col]));
                }
            }

            // Solve UX = Y
            for (int col = m - 1; col >= 0; col--) {
                bp[col] = bp[col].divide(lu[col][col]);
                final T bpCol = bp[col];
                for (int i = 0; i < col; i++) {
                    bp[i] = bp[i].subtract(bpCol.multiply(lu[i][col]));
                }
            }

            return new ArrayFieldVector<T>(bp, false);
        }

        
        public FieldMatrix<T> solve(FieldMatrix<T> b) {
            final int m = pivot.length;
            if (b.getRowDimension() != m) {
                throw new DimensionMismatchException(b.getRowDimension(), m);
            }
            if (singular) {
                throw new SingularMatrixException();
            }

            final int nColB = b.getColumnDimension();

            // Apply permutations to b
            final T[][] bp = MathArrays.buildArray(field, m, nColB);
            for (int row = 0; row < m; row++) {
                final T[] bpRow = bp[row];
                final int pRow = pivot[row];
                for (int col = 0; col < nColB; col++) {
                    bpRow[col] = b.getEntry(pRow, col);
                }
            }

            // Solve LY = b
            for (int col = 0; col < m; col++) {
                final T[] bpCol = bp[col];
                for (int i = col + 1; i < m; i++) {
                    final T[] bpI = bp[i];
                    final T luICol = lu[i][col];
                    for (int j = 0; j < nColB; j++) {
                        bpI[j] = bpI[j].subtract(bpCol[j].multiply(luICol));
                    }
                }
            }

            // Solve UX = Y
            for (int col = m - 1; col >= 0; col--) {
                final T[] bpCol = bp[col];
                final T luDiag = lu[col][col];
                for (int j = 0; j < nColB; j++) {
                    bpCol[j] = bpCol[j].divide(luDiag);
                }
                for (int i = 0; i < col; i++) {
                    final T[] bpI = bp[i];
                    final T luICol = lu[i][col];
                    for (int j = 0; j < nColB; j++) {
                        bpI[j] = bpI[j].subtract(bpCol[j].multiply(luICol));
                    }
                }
            }

            return new Array2DRowFieldMatrix<T>(field, bp, false);

        }

        
        public FieldMatrix<T> getInverse() {
            final int m = pivot.length;
            final T one = field.getOne();
            FieldMatrix<T> identity = new Array2DRowFieldMatrix<T>(field, m, m);
            for (int i = 0; i < m; ++i) {
                identity.setEntry(i, i, one);
            }
            return solve(identity);
        }
    }
}
