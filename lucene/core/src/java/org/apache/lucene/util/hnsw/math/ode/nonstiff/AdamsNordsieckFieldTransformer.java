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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowFieldMatrix;
import org.apache.lucene.util.hnsw.math.linear.ArrayFieldVector;
import org.apache.lucene.util.hnsw.math.linear.FieldDecompositionSolver;
import org.apache.lucene.util.hnsw.math.linear.FieldLUDecomposition;
import org.apache.lucene.util.hnsw.math.linear.FieldMatrix;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class AdamsNordsieckFieldTransformer<T extends RealFieldElement<T>> {

    
    private static final Map<Integer,
                         Map<Field<? extends RealFieldElement<?>>,
                                   AdamsNordsieckFieldTransformer<? extends RealFieldElement<?>>>> CACHE =
        new HashMap<Integer, Map<Field<? extends RealFieldElement<?>>,
                                 AdamsNordsieckFieldTransformer<? extends RealFieldElement<?>>>>();

    
    private final Field<T> field;

    
    private final Array2DRowFieldMatrix<T> update;

    
    private final T[] c1;

    
    private AdamsNordsieckFieldTransformer(final Field<T> field, final int n) {

        this.field = field;
        final int rows = n - 1;

        // compute coefficients
        FieldMatrix<T> bigP = buildP(rows);
        FieldDecompositionSolver<T> pSolver =
            new FieldLUDecomposition<T>(bigP).getSolver();

        T[] u = MathArrays.buildArray(field, rows);
        Arrays.fill(u, field.getOne());
        c1 = pSolver.solve(new ArrayFieldVector<T>(u, false)).toArray();

        // update coefficients are computed by combining transform from
        // Nordsieck to multistep, then shifting rows to represent step advance
        // then applying inverse transform
        T[][] shiftedP = bigP.getData();
        for (int i = shiftedP.length - 1; i > 0; --i) {
            // shift rows
            shiftedP[i] = shiftedP[i - 1];
        }
        shiftedP[0] = MathArrays.buildArray(field, rows);
        Arrays.fill(shiftedP[0], field.getZero());
        update = new Array2DRowFieldMatrix<T>(pSolver.solve(new Array2DRowFieldMatrix<T>(shiftedP, false)).getData());

    }

    
    @SuppressWarnings("unchecked")
    public static <T extends RealFieldElement<T>> AdamsNordsieckFieldTransformer<T>
    getInstance(final Field<T> field, final int nSteps) {
        synchronized(CACHE) {
            Map<Field<? extends RealFieldElement<?>>,
                      AdamsNordsieckFieldTransformer<? extends RealFieldElement<?>>> map = CACHE.get(nSteps);
            if (map == null) {
                map = new HashMap<Field<? extends RealFieldElement<?>>,
                                        AdamsNordsieckFieldTransformer<? extends RealFieldElement<?>>>();
                CACHE.put(nSteps, map);
            }
            @SuppressWarnings("rawtypes") // use rawtype to avoid compilation problems with java 1.5
            AdamsNordsieckFieldTransformer t = map.get(field);
            if (t == null) {
                t = new AdamsNordsieckFieldTransformer<T>(field, nSteps);
                map.put(field, (AdamsNordsieckFieldTransformer<T>) t);
            }
            return (AdamsNordsieckFieldTransformer<T>) t;

        }
    }

    
    private FieldMatrix<T> buildP(final int rows) {

        final T[][] pData = MathArrays.buildArray(field, rows, rows);

        for (int i = 1; i <= pData.length; ++i) {
            // build the P matrix elements from Taylor series formulas
            final T[] pI = pData[i - 1];
            final int factor = -i;
            T aj = field.getZero().add(factor);
            for (int j = 1; j <= pI.length; ++j) {
                pI[j - 1] = aj.multiply(j + 1);
                aj = aj.multiply(factor);
            }
        }

        return new Array2DRowFieldMatrix<T>(pData, false);

    }

    

    public Array2DRowFieldMatrix<T> initializeHighOrderDerivatives(final T h, final T[] t,
                                                                   final T[][] y,
                                                                   final T[][] yDot) {

        // using Taylor series with di = ti - t0, we get:
        //  y(ti)  - y(t0)  - di y'(t0) =   di^2 / h^2 s2 + ... +   di^k     / h^k sk + O(h^k)
        //  y'(ti) - y'(t0)             = 2 di   / h^2 s2 + ... + k di^(k-1) / h^k sk + O(h^(k-1))
        // we write these relations for i = 1 to i= 1+n/2 as a set of n + 2 linear
        // equations depending on the Nordsieck vector [s2 ... sk rk], so s2 to sk correspond
        // to the appropriately truncated Taylor expansion, and rk is the Taylor remainder.
        // The goal is to have s2 to sk as accurate as possible considering the fact the sum is
        // truncated and we don't want the error terms to be included in s2 ... sk, so we need
        // to solve also for the remainder
        final T[][] a     = MathArrays.buildArray(field, c1.length + 1, c1.length + 1);
        final T[][] b     = MathArrays.buildArray(field, c1.length + 1, y[0].length);
        final T[]   y0    = y[0];
        final T[]   yDot0 = yDot[0];
        for (int i = 1; i < y.length; ++i) {

            final T di    = t[i].subtract(t[0]);
            final T ratio = di.divide(h);
            T dikM1Ohk    = h.reciprocal();

            // linear coefficients of equations
            // y(ti) - y(t0) - di y'(t0) and y'(ti) - y'(t0)
            final T[] aI    = a[2 * i - 2];
            final T[] aDotI = (2 * i - 1) < a.length ? a[2 * i - 1] : null;
            for (int j = 0; j < aI.length; ++j) {
                dikM1Ohk = dikM1Ohk.multiply(ratio);
                aI[j]    = di.multiply(dikM1Ohk);
                if (aDotI != null) {
                    aDotI[j]  = dikM1Ohk.multiply(j + 2);
                }
            }

            // expected value of the previous equations
            final T[] yI    = y[i];
            final T[] yDotI = yDot[i];
            final T[] bI    = b[2 * i - 2];
            final T[] bDotI = (2 * i - 1) < b.length ? b[2 * i - 1] : null;
            for (int j = 0; j < yI.length; ++j) {
                bI[j]    = yI[j].subtract(y0[j]).subtract(di.multiply(yDot0[j]));
                if (bDotI != null) {
                    bDotI[j] = yDotI[j].subtract(yDot0[j]);
                }
            }

        }

        // solve the linear system to get the best estimate of the Nordsieck vector [s2 ... sk],
        // with the additional terms s(k+1) and c grabbing the parts after the truncated Taylor expansion
        final FieldLUDecomposition<T> decomposition = new FieldLUDecomposition<T>(new Array2DRowFieldMatrix<T>(a, false));
        final FieldMatrix<T> x = decomposition.getSolver().solve(new Array2DRowFieldMatrix<T>(b, false));

        // extract just the Nordsieck vector [s2 ... sk]
        final Array2DRowFieldMatrix<T> truncatedX =
                        new Array2DRowFieldMatrix<T>(field, x.getRowDimension() - 1, x.getColumnDimension());
        for (int i = 0; i < truncatedX.getRowDimension(); ++i) {
            for (int j = 0; j < truncatedX.getColumnDimension(); ++j) {
                truncatedX.setEntry(i, j, x.getEntry(i, j));
            }
        }
        return truncatedX;

    }

    
    public Array2DRowFieldMatrix<T> updateHighOrderDerivativesPhase1(final Array2DRowFieldMatrix<T> highOrder) {
        return update.multiply(highOrder);
    }

    
    public void updateHighOrderDerivativesPhase2(final T[] start,
                                                 final T[] end,
                                                 final Array2DRowFieldMatrix<T> highOrder) {
        final T[][] data = highOrder.getDataRef();
        for (int i = 0; i < data.length; ++i) {
            final T[] dataI = data[i];
            final T c1I = c1[i];
            for (int j = 0; j < dataI.length; ++j) {
                dataI[j] = dataI[j].add(c1I.multiply(start[j].subtract(end[j])));
            }
        }
    }

}
