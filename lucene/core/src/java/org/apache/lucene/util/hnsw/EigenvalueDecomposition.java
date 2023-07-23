package org.apache.lucene.util.hnsw;

import static org.apache.lucene.util.VectorUtil.dotProduct;

class EigenvalueDecomposition {

    private final float[][] eigenvectors;
    private final float[] eigenvalues;

    public EigenvalueDecomposition(float[][] matrix) {
        int n = matrix.length;

        eigenvalues = new float[n];
        eigenvectors = new float[n][n];

        float[][] A = copy(matrix);
        float[][] Q = new float[n][n];
        float[][] R = new float[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i < j) {
                    R[i][j] = A[i][j];
                } else {
                    R[i][j] = 0;
                }
            }
        }

        int iterations = 100;  // Maximum number of iterations.
        for (int k = 0; k < iterations; k++) {
            float shift = A[n - 1][n - 1];  // Rayleigh quotient shift.
            assert Float.isFinite(shift) : shift;

            for (int i = 0; i < n; i++) {
                A[i][i] -= shift;
            }
            qrDecomp(A, Q, R);
            for (int i = 0; i < n; i++) {
                A[i][i] += shift;
            }

            for (int i = 0; i < n; i++) {
                assert Float.isFinite(A[i][i]) : "Diagonal element in A is NaN after shift at iteration " + k;
            }

            A = multiply(R, Q);
            Q = multiply(Q, R);

            for (int i = 0; i < n; i++) {
                assert Float.isFinite(A[i][i]) : "Diagonal element in A is NaN after shift at iteration " + k;
            }

            // Check for convergence (in practice, you'd use a more robust check)
            float a = Math.abs(A[n - 1][n - 2]);
            assert Float.isFinite(a) : a;
            if (a < 1e-10) {
                break;
            }
        }

        computeEigen(A, Q);
    }

    static float[][] multiply(float[][] a, float[][] b) {
        int rowsA = a.length;
        int colsA = a[0].length;
        int rowsB = b.length;
        int colsB = b[0].length;

        if (colsA != rowsB) {
            throw new IllegalArgumentException("Number of columns in the first matrix must be equal to the number of rows in the second matrix.");
        }

        float[][] result = new float[rowsA][colsB];

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return result;
    }

    public float[][] getEigenvectors() {
        return eigenvectors;
    }

    public float[] getEigenvalues() {
        return eigenvalues;
    }

    static float[] project(float[] u, float[] v) {
        float scale = dotProduct(u, v) / dotProduct(v, v);
        return scale(v, scale);
    }

    static float[] orthogonalize(float[] u, float[] v) {
        float[] projection = project(u, v);
        return subtract(u, projection);
    }

    static float[] normalize(float[] u) {
        float norm = (float) Math.sqrt(dotProduct(u, u));
        if (norm != 0) {
            return scale(u, 1 / norm);
        }
        return u;
    }

    static float magnitude(float[] v) {
        return (float) Math.sqrt(dotProduct(v, v));
    }

    /**
     * Compute Q and R for QR decomposition of A such that A=QR.
     * Q and R should be matrices of the same size as A; their contents will be overwritten by the computation.
     * R's contents are ignored but
     */
    static void qrDecomp(float[][] A, float[][] Q, float[][] R) {
        int m = A.length;    // rows
        int n = A[0].length; // columns

        // Householder decomposition
        for (int k = 0; k < n; k++) {
            float[] x = getColumn(A, k);
            float[] e = new float[x.length];
            e[0] = 1;
            float[] v_k = subtract(x, scale(e, x[0] >= 0 ? -magnitude(x) : magnitude(x)));
            float[][] H_k = subtract(identity(m), scaleOuterProduct(v_k, v_k, 2 / dotProduct(v_k, v_k)));
            float[][] A_next = multiply(H_k, A);

            for (int i = 0; i < m; i++) {
                System.arraycopy(A_next[i], 0, A[i], 0, n);
            }

            updateColumn(Q, getColumn(H_k, k), k);
            updateColumn(R, getColumn(A, k), k);
        }
    }

    static float[][] subtract(float[][] mat1, float[][] mat2) {
        int rows = mat1.length;
        int cols = mat1[0].length;
        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = mat1[i][j] - mat2[i][j];
            }
        }

        return result;
    }

    static float[][] scaleOuterProduct(float[] u, float[] v, float scale) {
        int rows = u.length;
        int cols = v.length;
        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = u[i] * v[j] * scale;
            }
        }

        return result;
    }

    private static float[] copyVector(float[] orig) {
        float[] copy = new float[orig.length];
        System.arraycopy(orig, 0, copy, 0, orig.length);
        return copy;
    }

    private static float[] getColumn(float[][] mat, int columnIndex) {
        float[] column = new float[mat.length];
        for (int i = 0; i < mat.length; i++) {
            column[i] = mat[i][columnIndex];
        }
        return column;
    }

    private static void updateColumn(float[][] mat, float[] col, int colIndex) {
        for (int i = 0; i < mat.length; i++) {
            mat[i][colIndex] = col[i];
        }
    }

    private void computeEigen(float[][] A, float[][] R) {
        for (int i = 0; i < A.length; i++) {
            eigenvalues[i] = A[i][i];
            copyRow(eigenvectors, i, R[i]);
        }
    }

    static float[][] copy(float[][] orig) {
        float[][] copy = new float[orig.length][orig[0].length];
        for (int i = 0; i < orig.length; i++) {
            copy[i] = copyRow(orig, i);
        }
        return copy;
    }

    private static float[] copyRow(float[][] mat, int rowIndex) {
        float[] row = new float[mat[0].length];
        System.arraycopy(mat[rowIndex], 0, row, 0, row.length);
        return row;
    }

    private static void copyRow(float[][] dest, int rowIndex, float[] srcRow) {
        System.arraycopy(srcRow, 0, dest[rowIndex], 0, srcRow.length);
    }

    static float[][] identity(int n) {
        float[][] I = new float[n][n];
        for (int i = 0; i < n; i++) {
            I[i][i] = 1;
        }
        return I;
    }

    static float[] subtract(float[] vec1, float[] vec2) {
        float[] result = new float[vec1.length];
        for (int i = 0; i < vec1.length; i++) {
            result[i] = vec1[i] - vec2[i];
        }
        return result;
    }

    static float[] scale(float[] vec, float scale) {
        assert scale != 0;

        float[] result = new float[vec.length];
        for (int i = 0; i < vec.length; i++) {
            result[i] = vec[i] * scale;
        }
        return result;
    }
}
