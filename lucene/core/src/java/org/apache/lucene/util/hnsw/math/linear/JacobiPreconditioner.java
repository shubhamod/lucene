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

import org.apache.lucene.util.hnsw.math.analysis.function.Sqrt;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class JacobiPreconditioner extends RealLinearOperator {

    
    private final ArrayRealVector diag;

    
    public JacobiPreconditioner(final double[] diag, final boolean deep) {
        this.diag = new ArrayRealVector(diag, deep);
    }

    
    public static JacobiPreconditioner create(final RealLinearOperator a)
        throws NonSquareOperatorException {
        final int n = a.getColumnDimension();
        if (a.getRowDimension() != n) {
            throw new NonSquareOperatorException(a.getRowDimension(), n);
        }
        final double[] diag = new double[n];
        if (a instanceof AbstractRealMatrix) {
            final AbstractRealMatrix m = (AbstractRealMatrix) a;
            for (int i = 0; i < n; i++) {
                diag[i] = m.getEntry(i, i);
            }
        } else {
            final ArrayRealVector x = new ArrayRealVector(n);
            for (int i = 0; i < n; i++) {
                x.set(0.);
                x.setEntry(i, 1.);
                diag[i] = a.operate(x).getEntry(i);
            }
        }
        return new JacobiPreconditioner(diag, false);
    }

    
    @Override
    public int getColumnDimension() {
        return diag.getDimension();
    }

    
    @Override
    public int getRowDimension() {
        return diag.getDimension();
    }

    
    @Override
    public RealVector operate(final RealVector x) {
        // Dimension check is carried out by ebeDivide
        return new ArrayRealVector(MathArrays.ebeDivide(x.toArray(),
                                                        diag.toArray()),
                                   false);
    }

    
    public RealLinearOperator sqrt() {
        final RealVector sqrtDiag = diag.map(new Sqrt());
        return new RealLinearOperator() {
            
            @Override
            public RealVector operate(final RealVector x) {
                return new ArrayRealVector(MathArrays.ebeDivide(x.toArray(),
                                                                sqrtDiag.toArray()),
                                           false);
            }

            
            @Override
            public int getRowDimension() {
                return sqrtDiag.getDimension();
            }

            
            @Override
            public int getColumnDimension() {
                return sqrtDiag.getDimension();
            }
        };
    }
}
