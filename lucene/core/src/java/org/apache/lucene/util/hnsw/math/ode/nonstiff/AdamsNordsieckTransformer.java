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

package org.apache.lucene.util.hnsw.math.ode.nonstiff;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.util.hnsw.math.fraction.BigFraction;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowFieldMatrix;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.ArrayFieldVector;
import org.apache.lucene.util.hnsw.math.linear.FieldDecompositionSolver;
import org.apache.lucene.util.hnsw.math.linear.FieldLUDecomposition;
import org.apache.lucene.util.hnsw.math.linear.FieldMatrix;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.QRDecomposition;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;


public class AdamsNordsieckTransformer {

    
    private static final Map<Integer, AdamsNordsieckTransformer> CACHE =
        new HashMap<Integer, AdamsNordsieckTransformer>();

    
    private final Array2DRowRealMatrix update;

    
    private final double[] c1;

    
    private AdamsNordsieckTransformer(final int n) {

        final int rows = n - 1;

        // compute exact coefficients
        FieldMatrix<BigFraction> bigP = buildP(rows);
        FieldDecompositionSolver<BigFraction> pSolver =
            new FieldLUDecomposition<BigFraction>(bigP).getSolver();

        BigFraction[] u = new BigFraction[rows];
        Arrays.fill(u, BigFraction.ONE);
        BigFraction[] bigC1 = pSolver.solve(new ArrayFieldVector<BigFraction>(u, false)).toArray();

        // update coefficients are computed by combining transform from
        // Nordsieck to multistep, then shifting rows to represent step advance
        // then applying inverse transform
        BigFraction[][] shiftedP = bigP.getData();
        for (int i = shiftedP.length - 1; i > 0; --i) {
            // shift rows
            shiftedP[i] = shiftedP[i - 1];
        }
        shiftedP[0] = new BigFraction[rows];
        Arrays.fill(shiftedP[0], BigFraction.ZERO);
        FieldMatrix<BigFraction> bigMSupdate =
            pSolver.solve(new Array2DRowFieldMatrix<BigFraction>(shiftedP, false));

        // convert coefficients to double
        update         = MatrixUtils.bigFractionMatrixToRealMatrix(bigMSupdate);
        c1             = new double[rows];
        for (int i = 0; i < rows; ++i) {
            c1[i] = bigC1[i].doubleValue();
        }

    }

    
    public static AdamsNordsieckTransformer getInstance(final int nSteps) {
        synchronized(CACHE) {
            AdamsNordsieckTransformer t = CACHE.get(nSteps);
            if (t == null) {
                t = new AdamsNordsieckTransformer(nSteps);
                CACHE.put(nSteps, t);
            }
            return t;
        }
    }

    
    @Deprecated
    public int getNSteps() {
        return c1.length;
    }

    
    private FieldMatrix<BigFraction> buildP(final int rows) {

        final BigFraction[][] pData = new BigFraction[rows][rows];

        for (int i = 1; i <= pData.length; ++i) {
            // build the P matrix elements from Taylor series formulas
            final BigFraction[] pI = pData[i - 1];
            final int factor = -i;
            int aj = factor;
            for (int j = 1; j <= pI.length; ++j) {
                pI[j - 1] = new BigFraction(aj * (j + 1));
                aj *= factor;
            }
        }

        return new Array2DRowFieldMatrix<BigFraction>(pData, false);

    }

    

    public Array2DRowRealMatrix initializeHighOrderDerivatives(final double h, final double[] t,
                                                               final double[][] y,
                                                               final double[][] yDot) {

        // using Taylor series with di = ti - t0, we get:
        //  y(ti)  - y(t0)  - di y'(t0) =   di^2 / h^2 s2 + ... +   di^k     / h^k sk + O(h^k)
        //  y'(ti) - y'(t0)             = 2 di   / h^2 s2 + ... + k di^(k-1) / h^k sk + O(h^(k-1))
        // we write these relations for i = 1 to i= 1+n/2 as a set of n + 2 linear
        // equations depending on the Nordsieck vector [s2 ... sk rk], so s2 to sk correspond
        // to the appropriately truncated Taylor expansion, and rk is the Taylor remainder.
        // The goal is to have s2 to sk as accurate as possible considering the fact the sum is
        // truncated and we don't want the error terms to be included in s2 ... sk, so we need
        // to solve also for the remainder
        final double[][] a     = new double[c1.length + 1][c1.length + 1];
        final double[][] b     = new double[c1.length + 1][y[0].length];
        final double[]   y0    = y[0];
        final double[]   yDot0 = yDot[0];
        for (int i = 1; i < y.length; ++i) {

            final double di    = t[i] - t[0];
            final double ratio = di / h;
            double dikM1Ohk    =  1 / h;

            // linear coefficients of equations
            // y(ti) - y(t0) - di y'(t0) and y'(ti) - y'(t0)
            final double[] aI    = a[2 * i - 2];
            final double[] aDotI = (2 * i - 1) < a.length ? a[2 * i - 1] : null;
            for (int j = 0; j < aI.length; ++j) {
                dikM1Ohk *= ratio;
                aI[j]     = di      * dikM1Ohk;
                if (aDotI != null) {
                    aDotI[j]  = (j + 2) * dikM1Ohk;
                }
            }

            // expected value of the previous equations
            final double[] yI    = y[i];
            final double[] yDotI = yDot[i];
            final double[] bI    = b[2 * i - 2];
            final double[] bDotI = (2 * i - 1) < b.length ? b[2 * i - 1] : null;
            for (int j = 0; j < yI.length; ++j) {
                bI[j]    = yI[j] - y0[j] - di * yDot0[j];
                if (bDotI != null) {
                    bDotI[j] = yDotI[j] - yDot0[j];
                }
            }

        }

        // solve the linear system to get the best estimate of the Nordsieck vector [s2 ... sk],
        // with the additional terms s(k+1) and c grabbing the parts after the truncated Taylor expansion
        final QRDecomposition decomposition = new QRDecomposition(new Array2DRowRealMatrix(a, false));
        final RealMatrix x = decomposition.getSolver().solve(new Array2DRowRealMatrix(b, false));

        // extract just the Nordsieck vector [s2 ... sk]
        final Array2DRowRealMatrix truncatedX = new Array2DRowRealMatrix(x.getRowDimension() - 1, x.getColumnDimension());
        for (int i = 0; i < truncatedX.getRowDimension(); ++i) {
            for (int j = 0; j < truncatedX.getColumnDimension(); ++j) {
                truncatedX.setEntry(i, j, x.getEntry(i, j));
            }
        }
        return truncatedX;

    }

    
    public Array2DRowRealMatrix updateHighOrderDerivativesPhase1(final Array2DRowRealMatrix highOrder) {
        return update.multiply(highOrder);
    }

    
    public void updateHighOrderDerivativesPhase2(final double[] start,
                                                 final double[] end,
                                                 final Array2DRowRealMatrix highOrder) {
        final double[][] data = highOrder.getDataRef();
        for (int i = 0; i < data.length; ++i) {
            final double[] dataI = data[i];
            final double c1I = c1[i];
            for (int j = 0; j < dataI.length; ++j) {
                dataI[j] += c1I * (start[j] - end[j]);
            }
        }
    }

}
