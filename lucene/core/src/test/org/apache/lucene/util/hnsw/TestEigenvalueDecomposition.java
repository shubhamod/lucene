/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.util.hnsw;

import org.apache.lucene.tests.util.LuceneTestCase;

import static org.apache.lucene.util.VectorUtil.dotProduct;

public class TestEigenvalueDecomposition extends LuceneTestCase {
    public void testEigenvalueDecomposition() {
        float[][] matrix = new float[][]{
                {4, 1},
                {1, 3}
        };

        EigenvalueDecomposition eig = new EigenvalueDecomposition(matrix);
        float[][] eigenvectors = eig.getEigenvectors();
        float[] eigenvalues = eig.getEigenvalues();

        // Validate the eigenvalues.
        assertEquals(5.0f, eigenvalues[0], 1e-6);
        assertEquals(2.0f, eigenvalues[1], 1e-6);

        // Validate the eigenvectors. Note that eigenvectors can be scaled by any constant, so we should
        // only compare their direction, not their magnitude. We are assuming here that your
        // EigenvalueDecomposition implementation normalizes the eigenvectors to have length 1.
        assertEquals(Math.sqrt(2) / 2, Math.abs(eigenvectors[0][0]), 1e-6);
        assertEquals(Math.sqrt(2) / 2, Math.abs(eigenvectors[0][1]), 1e-6);
        assertEquals(Math.sqrt(2) / 2, Math.abs(eigenvectors[1][0]), 1e-6);
        assertEquals(Math.sqrt(2) / 2, Math.abs(eigenvectors[1][1]), 1e-6);
    }

    private static final float EPSILON = 1e-6f;

    public void testQrDecomp() {
        float[][] A = {
                {12, -51, 4},
                {6, 167, -68},
                {-4, 24, -41}
        };

        float[][] R = new float[A.length][A.length];

        float[][] Q = EigenvalueDecomposition.copy(A);
        EigenvalueDecomposition.qrDecomp(Q, R);

        // Check if R is upper triangular
        for (int i = 0; i < R.length; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(0, R[i][j], 1e-6);
            }
        }

        // Check if Q's columns are orthogonal
        for (int i = 0; i < Q[0].length; i++) {
            for (int j = i + 1; j < Q[0].length; j++) {
                float[] column1 = getColumn(Q, i);
                float[] column2 = getColumn(Q, j);
                float dotProduct = dotProduct(column1, column2);
                assertEquals(0, dotProduct, 1e-6);
            }
        }

        // check that the decomposition product is the original matrix
        float[][] QR = EigenvalueDecomposition.multiply(Q, R);
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[i].length; j++) {
                assertEquals(A[i][j], QR[i][j], EPSILON);
            }
        }
    }

    private float[] getColumn(float[][] matrix, int column) {
        float[] result = new float[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = matrix[i][column];
        }
        return result;
    }

    public void testScale() {
        float[] v = {1, 2, 3};
        float scalar = 2;
        float[] expected = {2, 4, 6};
        float[] result = EigenvalueDecomposition.scale(v, scalar);
        assertArrayEquals(expected, result, EPSILON);
    }

    public void testSubtract() {
        float[] v1 = {1, 2, 3};
        float[] v2 = {4, 5, 6};
        float[] expected = {-3, -3, -3};
        float[] result = EigenvalueDecomposition.subtract(v1, v2);
        assertArrayEquals(expected, result, EPSILON);
    }

    public void testIdentityMatrix() {
        int n = 3;
        float[][] expected = {
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        };
        float[][] result = EigenvalueDecomposition.identity(n);
        assertArrayEquals(expected, result);
    }

    public void testMultiply() {
        float[][] a = {
                {1, 2},
                {3, 4}
        };
        float[][] b = {
                {5, 6},
                {7, 8}
        };
        float[][] expected = {
                {19, 22},
                {43, 50}
        };
        float[][] result = EigenvalueDecomposition.multiply(a, b);
        assertArrayEquals(expected, result);
    }
}

