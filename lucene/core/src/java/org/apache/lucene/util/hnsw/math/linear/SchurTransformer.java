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

import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;


class SchurTransformer {
    
    private static final int MAX_ITERATIONS = 100;

    
    private final double matrixP[][];
    
    private final double matrixT[][];
    
    private RealMatrix cachedP;
    
    private RealMatrix cachedT;
    
    private RealMatrix cachedPt;

    
    private final double epsilon = Precision.EPSILON;

    
    SchurTransformer(final RealMatrix matrix) {
        if (!matrix.isSquare()) {
            throw new NonSquareMatrixException(matrix.getRowDimension(),
                                               matrix.getColumnDimension());
        }

        HessenbergTransformer transformer = new HessenbergTransformer(matrix);
        matrixT = transformer.getH().getData();
        matrixP = transformer.getP().getData();
        cachedT = null;
        cachedP = null;
        cachedPt = null;

        // transform matrix
        transform();
    }

    
    public RealMatrix getP() {
        if (cachedP == null) {
            cachedP = MatrixUtils.createRealMatrix(matrixP);
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

    
    public RealMatrix getT() {
        if (cachedT == null) {
            cachedT = MatrixUtils.createRealMatrix(matrixT);
        }

        // return the cached matrix
        return cachedT;
    }

    
    private void transform() {
        final int n = matrixT.length;

        // compute matrix norm
        final double norm = getNorm();

        // shift information
        final ShiftInfo shift = new ShiftInfo();

        // Outer loop over eigenvalue index
        int iteration = 0;
        int iu = n - 1;
        while (iu >= 0) {

            // Look for single small sub-diagonal element
            final int il = findSmallSubDiagonalElement(iu, norm);

            // Check for convergence
            if (il == iu) {
                // One root found
                matrixT[iu][iu] += shift.exShift;
                iu--;
                iteration = 0;
            } else if (il == iu - 1) {
                // Two roots found
                double p = (matrixT[iu - 1][iu - 1] - matrixT[iu][iu]) / 2.0;
                double q = p * p + matrixT[iu][iu - 1] * matrixT[iu - 1][iu];
                matrixT[iu][iu] += shift.exShift;
                matrixT[iu - 1][iu - 1] += shift.exShift;

                if (q >= 0) {
                    double z = FastMath.sqrt(FastMath.abs(q));
                    if (p >= 0) {
                        z = p + z;
                    } else {
                        z = p - z;
                    }
                    final double x = matrixT[iu][iu - 1];
                    final double s = FastMath.abs(x) + FastMath.abs(z);
                    p = x / s;
                    q = z / s;
                    final double r = FastMath.sqrt(p * p + q * q);
                    p /= r;
                    q /= r;

                    // Row modification
                    for (int j = iu - 1; j < n; j++) {
                        z = matrixT[iu - 1][j];
                        matrixT[iu - 1][j] = q * z + p * matrixT[iu][j];
                        matrixT[iu][j] = q * matrixT[iu][j] - p * z;
                    }

                    // Column modification
                    for (int i = 0; i <= iu; i++) {
                        z = matrixT[i][iu - 1];
                        matrixT[i][iu - 1] = q * z + p * matrixT[i][iu];
                        matrixT[i][iu] = q * matrixT[i][iu] - p * z;
                    }

                    // Accumulate transformations
                    for (int i = 0; i <= n - 1; i++) {
                        z = matrixP[i][iu - 1];
                        matrixP[i][iu - 1] = q * z + p * matrixP[i][iu];
                        matrixP[i][iu] = q * matrixP[i][iu] - p * z;
                    }
                }
                iu -= 2;
                iteration = 0;
            } else {
                // No convergence yet
                computeShift(il, iu, iteration, shift);

                // stop transformation after too many iterations
                if (++iteration > MAX_ITERATIONS) {
                    throw new MaxCountExceededException(LocalizedFormats.CONVERGENCE_FAILED,
                                                        MAX_ITERATIONS);
                }

                // the initial houseHolder vector for the QR step
                final double[] hVec = new double[3];

                final int im = initQRStep(il, iu, shift, hVec);
                performDoubleQRStep(il, im, iu, shift, hVec);
            }
        }
    }

    
    private double getNorm() {
        double norm = 0.0;
        for (int i = 0; i < matrixT.length; i++) {
            // as matrix T is (quasi-)triangular, also take the sub-diagonal element into account
            for (int j = FastMath.max(i - 1, 0); j < matrixT.length; j++) {
                norm += FastMath.abs(matrixT[i][j]);
            }
        }
        return norm;
    }

    
    private int findSmallSubDiagonalElement(final int startIdx, final double norm) {
        int l = startIdx;
        while (l > 0) {
            double s = FastMath.abs(matrixT[l - 1][l - 1]) + FastMath.abs(matrixT[l][l]);
            if (s == 0.0) {
                s = norm;
            }
            if (FastMath.abs(matrixT[l][l - 1]) < epsilon * s) {
                break;
            }
            l--;
        }
        return l;
    }

    
    private void computeShift(final int l, final int idx, final int iteration, final ShiftInfo shift) {
        // Form shift
        shift.x = matrixT[idx][idx];
        shift.y = shift.w = 0.0;
        if (l < idx) {
            shift.y = matrixT[idx - 1][idx - 1];
            shift.w = matrixT[idx][idx - 1] * matrixT[idx - 1][idx];
        }

        // Wilkinson's original ad hoc shift
        if (iteration == 10) {
            shift.exShift += shift.x;
            for (int i = 0; i <= idx; i++) {
                matrixT[i][i] -= shift.x;
            }
            final double s = FastMath.abs(matrixT[idx][idx - 1]) + FastMath.abs(matrixT[idx - 1][idx - 2]);
            shift.x = 0.75 * s;
            shift.y = 0.75 * s;
            shift.w = -0.4375 * s * s;
        }

        // MATLAB's new ad hoc shift
        if (iteration == 30) {
            double s = (shift.y - shift.x) / 2.0;
            s = s * s + shift.w;
            if (s > 0.0) {
                s = FastMath.sqrt(s);
                if (shift.y < shift.x) {
                    s = -s;
                }
                s = shift.x - shift.w / ((shift.y - shift.x) / 2.0 + s);
                for (int i = 0; i <= idx; i++) {
                    matrixT[i][i] -= s;
                }
                shift.exShift += s;
                shift.x = shift.y = shift.w = 0.964;
            }
        }
    }

    
    private int initQRStep(int il, final int iu, final ShiftInfo shift, double[] hVec) {
        // Look for two consecutive small sub-diagonal elements
        int im = iu - 2;
        while (im >= il) {
            final double z = matrixT[im][im];
            final double r = shift.x - z;
            double s = shift.y - z;
            hVec[0] = (r * s - shift.w) / matrixT[im + 1][im] + matrixT[im][im + 1];
            hVec[1] = matrixT[im + 1][im + 1] - z - r - s;
            hVec[2] = matrixT[im + 2][im + 1];

            if (im == il) {
                break;
            }

            final double lhs = FastMath.abs(matrixT[im][im - 1]) * (FastMath.abs(hVec[1]) + FastMath.abs(hVec[2]));
            final double rhs = FastMath.abs(hVec[0]) * (FastMath.abs(matrixT[im - 1][im - 1]) +
                                                        FastMath.abs(z) +
                                                        FastMath.abs(matrixT[im + 1][im + 1]));

            if (lhs < epsilon * rhs) {
                break;
            }
            im--;
        }

        return im;
    }

    
    private void performDoubleQRStep(final int il, final int im, final int iu,
                                     final ShiftInfo shift, final double[] hVec) {

        final int n = matrixT.length;
        double p = hVec[0];
        double q = hVec[1];
        double r = hVec[2];

        for (int k = im; k <= iu - 1; k++) {
            boolean notlast = k != (iu - 1);
            if (k != im) {
                p = matrixT[k][k - 1];
                q = matrixT[k + 1][k - 1];
                r = notlast ? matrixT[k + 2][k - 1] : 0.0;
                shift.x = FastMath.abs(p) + FastMath.abs(q) + FastMath.abs(r);
                if (Precision.equals(shift.x, 0.0, epsilon)) {
                    continue;
                }
                p /= shift.x;
                q /= shift.x;
                r /= shift.x;
            }
            double s = FastMath.sqrt(p * p + q * q + r * r);
            if (p < 0.0) {
                s = -s;
            }
            if (s != 0.0) {
                if (k != im) {
                    matrixT[k][k - 1] = -s * shift.x;
                } else if (il != im) {
                    matrixT[k][k - 1] = -matrixT[k][k - 1];
                }
                p += s;
                shift.x = p / s;
                shift.y = q / s;
                double z = r / s;
                q /= p;
                r /= p;

                // Row modification
                for (int j = k; j < n; j++) {
                    p = matrixT[k][j] + q * matrixT[k + 1][j];
                    if (notlast) {
                        p += r * matrixT[k + 2][j];
                        matrixT[k + 2][j] -= p * z;
                    }
                    matrixT[k][j] -= p * shift.x;
                    matrixT[k + 1][j] -= p * shift.y;
                }

                // Column modification
                for (int i = 0; i <= FastMath.min(iu, k + 3); i++) {
                    p = shift.x * matrixT[i][k] + shift.y * matrixT[i][k + 1];
                    if (notlast) {
                        p += z * matrixT[i][k + 2];
                        matrixT[i][k + 2] -= p * r;
                    }
                    matrixT[i][k] -= p;
                    matrixT[i][k + 1] -= p * q;
                }

                // Accumulate transformations
                final int high = matrixT.length - 1;
                for (int i = 0; i <= high; i++) {
                    p = shift.x * matrixP[i][k] + shift.y * matrixP[i][k + 1];
                    if (notlast) {
                        p += z * matrixP[i][k + 2];
                        matrixP[i][k + 2] -= p * r;
                    }
                    matrixP[i][k] -= p;
                    matrixP[i][k + 1] -= p * q;
                }
            }  // (s != 0)
        }  // k loop

        // clean up pollution due to round-off errors
        for (int i = im + 2; i <= iu; i++) {
            matrixT[i][i-2] = 0.0;
            if (i > im + 2) {
                matrixT[i][i-3] = 0.0;
            }
        }
    }

    
    private static class ShiftInfo {
        // CHECKSTYLE: stop all

        
        double x;
        
        double y;
        
        double w;
        
        double exShift;

        // CHECKSTYLE: resume all
    }
}
