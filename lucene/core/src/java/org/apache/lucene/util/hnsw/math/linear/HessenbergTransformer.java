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

import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;


class HessenbergTransformer {
    
    private final double householderVectors[][];
    
    private final double ort[];
    
    private RealMatrix cachedP;
    
    private RealMatrix cachedPt;
    
    private RealMatrix cachedH;

    
    HessenbergTransformer(final RealMatrix matrix) {
        if (!matrix.isSquare()) {
            throw new NonSquareMatrixException(matrix.getRowDimension(),
                    matrix.getColumnDimension());
        }

        final int m = matrix.getRowDimension();
        householderVectors = matrix.getData();
        ort = new double[m];
        cachedP = null;
        cachedPt = null;
        cachedH = null;

        // transform matrix
        transform();
    }

    
    public RealMatrix getP() {
        if (cachedP == null) {
            final int n = householderVectors.length;
            final int high = n - 1;
            final double[][] pa = new double[n][n];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    pa[i][j] = (i == j) ? 1 : 0;
                }
            }

            for (int m = high - 1; m >= 1; m--) {
                if (householderVectors[m][m - 1] != 0.0) {
                    for (int i = m + 1; i <= high; i++) {
                        ort[i] = householderVectors[i][m - 1];
                    }

                    for (int j = m; j <= high; j++) {
                        double g = 0.0;

                        for (int i = m; i <= high; i++) {
                            g += ort[i] * pa[i][j];
                        }

                        // Double division avoids possible underflow
                        g = (g / ort[m]) / householderVectors[m][m - 1];

                        for (int i = m; i <= high; i++) {
                            pa[i][j] += g * ort[i];
                        }
                    }
                }
            }

            cachedP = MatrixUtils.createRealMatrix(pa);
        }
        return cachedP;
    }

    
    public RealMatrix getPT() {
        if (cachedPt == null) {
            cachedPt = getP().transpose();
        }

        // return the cached matrix
        return cachedPt;
    }

    
    public RealMatrix getH() {
        if (cachedH == null) {
            final int m = householderVectors.length;
            final double[][] h = new double[m][m];
            for (int i = 0; i < m; ++i) {
                if (i > 0) {
                    // copy the entry of the lower sub-diagonal
                    h[i][i - 1] = householderVectors[i][i - 1];
                }

                // copy upper triangular part of the matrix
                for (int j = i; j < m; ++j) {
                    h[i][j] = householderVectors[i][j];
                }
            }
            cachedH = MatrixUtils.createRealMatrix(h);
        }

        // return the cached matrix
        return cachedH;
    }

    
    double[][] getHouseholderVectorsRef() {
        return householderVectors;
    }

    
    private void transform() {
        final int n = householderVectors.length;
        final int high = n - 1;

        for (int m = 1; m <= high - 1; m++) {
            // Scale column.
            double scale = 0;
            for (int i = m; i <= high; i++) {
                scale += FastMath.abs(householderVectors[i][m - 1]);
            }

            if (!Precision.equals(scale, 0)) {
                // Compute Householder transformation.
                double h = 0;
                for (int i = high; i >= m; i--) {
                    ort[i] = householderVectors[i][m - 1] / scale;
                    h += ort[i] * ort[i];
                }
                final double g = (ort[m] > 0) ? -FastMath.sqrt(h) : FastMath.sqrt(h);

                h -= ort[m] * g;
                ort[m] -= g;

                // Apply Householder similarity transformation
                // H = (I - u*u' / h) * H * (I - u*u' / h)

                for (int j = m; j < n; j++) {
                    double f = 0;
                    for (int i = high; i >= m; i--) {
                        f += ort[i] * householderVectors[i][j];
                    }
                    f /= h;
                    for (int i = m; i <= high; i++) {
                        householderVectors[i][j] -= f * ort[i];
                    }
                }

                for (int i = 0; i <= high; i++) {
                    double f = 0;
                    for (int j = high; j >= m; j--) {
                        f += ort[j] * householderVectors[i][j];
                    }
                    f /= h;
                    for (int j = m; j <= high; j++) {
                        householderVectors[i][j] -= f * ort[j];
                    }
                }

                ort[m] = scale * ort[m];
                householderVectors[m][m - 1] = scale * g;
            }
        }
    }
}
