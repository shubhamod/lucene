package org.apache.lucene.util.hnsw;

import static org.apache.lucene.util.VectorUtil.dotProduct;

class EigenvalueDecomposition {
    private float[][] eigenvectors;
    private float[] eigenvalues;

    public EigenvalueDecomposition(float[][] matrix) {
        final int MAX_ITERATIONS = 100;
        eigenvectors = identity(matrix.length);
        eigenvalues = new float[matrix.length];

        float[][] A = matrix;
        float[][] Q = null;
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            Q = identity(A.length);
            float[][] R = new float[A.length][A.length];
            qrDecomp(A, Q, R);
            A = multiply(R, Q);
            eigenvectors = multiply(eigenvectors, Q);
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
     */
    static void qrDecomp(float[][] A, float[][] Q, float[][] R) {
        int m = A.length;    // rows
        int n = A[0].length; // columns

        // Modified Gram-Schmidt
        for (int i = 0; i < n; i++) {
            float[] ai = getColumn(A, i);
            float[] ui = copyVector(ai);

            for (int j = 0; j < i; j++) {
                float[] qj = getColumn(Q, j);
                float proj = dotProduct(ai, qj); // Project on already-orthogonalized vectors
                R[j][i] = proj; // Fill the upper-triangular matrix R
                ui = subtract(ui, scale(qj, proj)); // Orthogonalize with respect to already-orthogonalized vectors
            }

            R[i][i] = magnitude(ui);
            insertColumn(Q, normalize(ui), i);
        }
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

    private static void insertColumn(float[][] mat, float[] col, int colIndex) {
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
