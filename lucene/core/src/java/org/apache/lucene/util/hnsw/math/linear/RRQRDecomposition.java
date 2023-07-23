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



public class RRQRDecomposition extends QRDecomposition {

    
    private int[] p;

    
    private RealMatrix cachedP;


    
    public RRQRDecomposition(RealMatrix matrix) {
        this(matrix, 0d);
    }

   
    public RRQRDecomposition(RealMatrix matrix,  double threshold) {
        super(matrix, threshold);
    }

    
    @Override
    protected void decompose(double[][] qrt) {
        p = new int[qrt.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = i;
        }
        super.decompose(qrt);
    }

    
    @Override
    protected void performHouseholderReflection(int minor, double[][] qrt) {

        double l2NormSquaredMax = 0;
        // Find the unreduced column with the greatest L2-Norm
        int l2NormSquaredMaxIndex = minor;
        for (int i = minor; i < qrt.length; i++) {
            double l2NormSquared = 0;
            for (int j = 0; j < qrt[i].length; j++) {
                l2NormSquared += qrt[i][j] * qrt[i][j];
            }
            if (l2NormSquared > l2NormSquaredMax) {
                l2NormSquaredMax = l2NormSquared;
                l2NormSquaredMaxIndex = i;
            }
        }
        // swap the current column with that with the greated L2-Norm and record in p
        if (l2NormSquaredMaxIndex != minor) {
            double[] tmp1 = qrt[minor];
            qrt[minor] = qrt[l2NormSquaredMaxIndex];
            qrt[l2NormSquaredMaxIndex] = tmp1;
            int tmp2 = p[minor];
            p[minor] = p[l2NormSquaredMaxIndex];
            p[l2NormSquaredMaxIndex] = tmp2;
        }

        super.performHouseholderReflection(minor, qrt);

    }


    
    public RealMatrix getP() {
        if (cachedP == null) {
            int n = p.length;
            cachedP = MatrixUtils.createRealMatrix(n,n);
            for (int i = 0; i < n; i++) {
                cachedP.setEntry(p[i], i, 1);
            }
        }
        return cachedP ;
    }

    
    public int getRank(final double dropThreshold) {
        RealMatrix r    = getR();
        int rows        = r.getRowDimension();
        int columns     = r.getColumnDimension();
        int rank        = 1;
        double lastNorm = r.getFrobeniusNorm();
        double rNorm    = lastNorm;
        while (rank < FastMath.min(rows, columns)) {
            double thisNorm = r.getSubMatrix(rank, rows - 1, rank, columns - 1).getFrobeniusNorm();
            if (thisNorm == 0 || (thisNorm / lastNorm) * rNorm < dropThreshold) {
                break;
            }
            lastNorm = thisNorm;
            rank++;
        }
        return rank;
    }

    
    @Override
    public DecompositionSolver getSolver() {
        return new Solver(super.getSolver(), this.getP());
    }

    
    private static class Solver implements DecompositionSolver {

        
        private final DecompositionSolver upper;

        
        private RealMatrix p;

        
        private Solver(final DecompositionSolver upper, final RealMatrix p) {
            this.upper = upper;
            this.p     = p;
        }

        
        public boolean isNonSingular() {
            return upper.isNonSingular();
        }

        
        public RealVector solve(RealVector b) {
            return p.operate(upper.solve(b));
        }

        
        public RealMatrix solve(RealMatrix b) {
            return p.multiply(upper.solve(b));
        }

        
        public RealMatrix getInverse() {
            return solve(MatrixUtils.createRealIdentityMatrix(p.getRowDimension()));
        }
    }
}
