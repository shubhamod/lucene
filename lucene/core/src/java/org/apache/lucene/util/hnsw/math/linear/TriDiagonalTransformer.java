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

import org.apache.lucene.util.hnsw.math.util.FastMath;



class TriDiagonalTransformer {
    
    private final double householderVectors[][];
    
    private final double[] main;
    
    private final double[] secondary;
    
    private RealMatrix cachedQ;
    
    private RealMatrix cachedQt;
    
    private RealMatrix cachedT;

    
    TriDiagonalTransformer(RealMatrix matrix) {
        if (!matrix.isSquare()) {
            throw new NonSquareMatrixException(matrix.getRowDimension(),
                                               matrix.getColumnDimension());
        }

        final int m = matrix.getRowDimension();
        householderVectors = matrix.getData();
        main      = new double[m];
        secondary = new double[m - 1];
        cachedQ   = null;
        cachedQt  = null;
        cachedT   = null;

        // transform matrix
        transform();
    }

    
    public RealMatrix getQ() {
        if (cachedQ == null) {
            cachedQ = getQT().transpose();
        }
        return cachedQ;
    }

    
    public RealMatrix getQT() {
        if (cachedQt == null) {
            final int m = householderVectors.length;
            double[][] qta = new double[m][m];

            // build up first part of the matrix by applying Householder transforms
            for (int k = m - 1; k >= 1; --k) {
                final double[] hK = householderVectors[k - 1];
                qta[k][k] = 1;
                if (hK[k] != 0.0) {
                    final double inv = 1.0 / (secondary[k - 1] * hK[k]);
                    double beta = 1.0 / secondary[k - 1];
                    qta[k][k] = 1 + beta * hK[k];
                    for (int i = k + 1; i < m; ++i) {
                        qta[k][i] = beta * hK[i];
                    }
                    for (int j = k + 1; j < m; ++j) {
                        beta = 0;
                        for (int i = k + 1; i < m; ++i) {
                            beta += qta[j][i] * hK[i];
                        }
                        beta *= inv;
                        qta[j][k] = beta * hK[k];
                        for (int i = k + 1; i < m; ++i) {
                            qta[j][i] += beta * hK[i];
                        }
                    }
                }
            }
            qta[0][0] = 1;
            cachedQt = MatrixUtils.createRealMatrix(qta);
        }

        // return the cached matrix
        return cachedQt;
    }

    
    public RealMatrix getT() {
        if (cachedT == null) {
            final int m = main.length;
            double[][] ta = new double[m][m];
            for (int i = 0; i < m; ++i) {
                ta[i][i] = main[i];
                if (i > 0) {
                    ta[i][i - 1] = secondary[i - 1];
                }
                if (i < main.length - 1) {
                    ta[i][i + 1] = secondary[i];
                }
            }
            cachedT = MatrixUtils.createRealMatrix(ta);
        }

        // return the cached matrix
        return cachedT;
    }

    
    double[][] getHouseholderVectorsRef() {
        return householderVectors;
    }

    
    double[] getMainDiagonalRef() {
        return main;
    }

    
    double[] getSecondaryDiagonalRef() {
        return secondary;
    }

    
    private void transform() {
        final int m = householderVectors.length;
        final double[] z = new double[m];
        for (int k = 0; k < m - 1; k++) {

            //zero-out a row and a column simultaneously
            final double[] hK = householderVectors[k];
            main[k] = hK[k];
            double xNormSqr = 0;
            for (int j = k + 1; j < m; ++j) {
                final double c = hK[j];
                xNormSqr += c * c;
            }
            final double a = (hK[k + 1] > 0) ? -FastMath.sqrt(xNormSqr) : FastMath.sqrt(xNormSqr);
            secondary[k] = a;
            if (a != 0.0) {
                // apply Householder transform from left and right simultaneously

                hK[k + 1] -= a;
                final double beta = -1 / (a * hK[k + 1]);

                // compute a = beta A v, where v is the Householder vector
                // this loop is written in such a way
                //   1) only the upper triangular part of the matrix is accessed
                //   2) access is cache-friendly for a matrix stored in rows
                Arrays.fill(z, k + 1, m, 0);
                for (int i = k + 1; i < m; ++i) {
                    final double[] hI = householderVectors[i];
                    final double hKI = hK[i];
                    double zI = hI[i] * hKI;
                    for (int j = i + 1; j < m; ++j) {
                        final double hIJ = hI[j];
                        zI   += hIJ * hK[j];
                        z[j] += hIJ * hKI;
                    }
                    z[i] = beta * (z[i] + zI);
                }

                // compute gamma = beta vT z / 2
                double gamma = 0;
                for (int i = k + 1; i < m; ++i) {
                    gamma += z[i] * hK[i];
                }
                gamma *= beta / 2;

                // compute z = z - gamma v
                for (int i = k + 1; i < m; ++i) {
                    z[i] -= gamma * hK[i];
                }

                // update matrix: A = A - v zT - z vT
                // only the upper triangular part of the matrix is updated
                for (int i = k + 1; i < m; ++i) {
                    final double[] hI = householderVectors[i];
                    for (int j = i; j < m; ++j) {
                        hI[j] -= hK[i] * z[j] + z[i] * hK[j];
                    }
                }
            }
        }
        main[m - 1] = householderVectors[m - 1][m - 1];
    }
}
