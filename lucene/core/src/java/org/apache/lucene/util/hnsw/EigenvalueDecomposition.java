package org.apache.lucene.util.hnsw;

import org.apache.lucene.util.VectorUtil;

import static org.apache.lucene.util.VectorUtil.dotProduct;

class EigenvalueDecomposition {

    private final float[][] eigenvectors;
    private final float[] eigenvalues;

    public EigenvalueDecomposition(float[][] matrix) {
        int n = matrix.length;

        eigenvalues = new float[n];
        eigenvectors = new float[n][n];

        float[][] A = copy(matrix);
        float[][] Q = identity(n);
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
            qrDecomp(A, R);
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

    /**
     * Perform QR decomposition of A.  The A parameter will be modified in-place to compute Q.
     */
    static void qrDecomp(float[][] A, float[][] R) {
        for (int i = 0; i < A.length; i++) {
            float[] u = A[i];

            // Modified Gram-Schmidt orthogonalization
            for (int j = 0; j < i; j++) {
                float[] v = A[j];
                float r = dotProduct(u, v);
                R[j][i] = r;
                u = subtract(u, scale(v, r));
            }

            R[i][i] = (float) Math.sqrt(dotProduct(u, u));
            assert Float.isFinite(R[i][i]) : R[i][i];
            if (R[i][i] != 0) {
                u = scale(u, 1 / R[i][i]);
            }

            copyRow(A, i, u);
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
