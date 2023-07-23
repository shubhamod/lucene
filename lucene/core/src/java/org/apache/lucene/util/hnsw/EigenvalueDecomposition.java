package org.apache.lucene.util.hnsw;

import java.util.Arrays;

import static org.apache.lucene.util.VectorUtil.dotProduct;

class EigenvalueDecomposition {

    private final float[][] eigenvectors = null;
    private final float[] eigenvalues = null;

    public EigenvalueDecomposition(float[][] matrix) {
        // TODO
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
     * R must be an empty (all zeros) matrix of the same size as A; its contents will be overwritten by the computation.
     * Q will be returned by the method.
     * A must be a square matrix.
     */
    static float[][] qrDecomp(float[][] A, float[][] R) {
        int m = A.length;    // rows
        int n = A[0].length; // columns
        if (m != n) {
            throw new IllegalArgumentException("A matrix must be square, found " + m + "x" + n);
        }

        float[][] P;

        // Initialize Q as an identity matrix
        float[][] Q = identity(m);

        // Householder transformation
        for (int i = 0; i < n; i++) {
            float[] ai = getColumn(A, i);
            float alpha = computeAlpha(ai, i);
            float[] v = createV(ai, alpha, m, i);
            P = createP(m, v);
            A = multiply(P, A);
            Q = multiply(Q, P);
        }

        // Copy upper triangle of [the ending, modified] A to R
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], i, R[i], i, n - i);
        }

        return Q;
    }

    static float computeAlpha(float[] ai, int i) {
        // Create a subvector from the i-th position to the end
        float[] subAi = Arrays.copyOfRange(ai, i, ai.length);

        // Compute the norm of the subvector
        float norm = magnitude(subAi);

        // Multiply the norm by the signum of the first component of the subvector
        return -Math.signum(subAi[0]) * norm;
    }

    static float[] createV(float[] ai, float alpha, int m, int i) {
        float[] ei = identityColumn(m, i);
        float[] u = subtract(ai, scale(ei, alpha));
        return scale(u, 1 / magnitude(u));
    }

    static float[][] createP(int m, float[] v) {
        float[][] P = identity(m);
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
                P[j][k] -= 2 * v[j] * v[k];
            }
        }
        return P;
    }

    private static float[][] transpose(float[][] mat) {
        int rows = mat.length;
        int cols = mat[0].length;
        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = mat[i][j];
            }
        }
        return transposed;
    }

    private static float[] identityColumn(int size, int pos) {
        float[] column = new float[size];
        column[pos] = 1;
        return column;
    }

    static float[] getColumn(float[][] mat, int columnIndex) {
        float[] column = new float[mat.length];
        for (int i = 0; i < mat.length; i++) {
            column[i] = mat[i][columnIndex];
        }
        return column;
    }

    private void computeEigen(float[][] A, float[][] R) {
        for (int i = 0; i < A.length; i++) {
            eigenvalues[i] = A[i][i];
            copyRow(eigenvectors, i, R[i]);
        }
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
