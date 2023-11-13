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
package org.apache.lucene.util.hnsw.math.stat.regression;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.LUDecomposition;
import org.apache.lucene.util.hnsw.math.linear.QRDecomposition;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.stat.StatUtils;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.SecondMoment;


public class OLSMultipleLinearRegression extends AbstractMultipleLinearRegression {

    
    private QRDecomposition qr = null;

    
    private final double threshold;

    
    public OLSMultipleLinearRegression() {
        this(0d);
    }

    
    public OLSMultipleLinearRegression(final double threshold) {
        this.threshold = threshold;
    }

    
    public void newSampleData(double[] y, double[][] x) throws MathIllegalArgumentException {
        validateSampleData(x, y);
        newYSampleData(y);
        newXSampleData(x);
    }

    
    @Override
    public void newSampleData(double[] data, int nobs, int nvars) {
        super.newSampleData(data, nobs, nvars);
        qr = new QRDecomposition(getX(), threshold);
    }

    
    public RealMatrix calculateHat() {
        // Create augmented identity matrix
        RealMatrix Q = qr.getQ();
        final int p = qr.getR().getColumnDimension();
        final int n = Q.getColumnDimension();
        // No try-catch or advertised NotStrictlyPositiveException - NPE above if n < 3
        Array2DRowRealMatrix augI = new Array2DRowRealMatrix(n, n);
        double[][] augIData = augI.getDataRef();
        for (int i = 0; i < n; i++) {
            for (int j =0; j < n; j++) {
                if (i == j && i < p) {
                    augIData[i][j] = 1d;
                } else {
                    augIData[i][j] = 0d;
                }
            }
        }

        // Compute and return Hat matrix
        // No DME advertised - args valid if we get here
        return Q.multiply(augI).multiply(Q.transpose());
    }

    
    public double calculateTotalSumOfSquares() {
        if (isNoIntercept()) {
            return StatUtils.sumSq(getY().toArray());
        } else {
            return new SecondMoment().evaluate(getY().toArray());
        }
    }

    
    public double calculateResidualSumOfSquares() {
        final RealVector residuals = calculateResiduals();
        // No advertised DME, args are valid
        return residuals.dotProduct(residuals);
    }

    
    public double calculateRSquared() {
        return 1 - calculateResidualSumOfSquares() / calculateTotalSumOfSquares();
    }

    
    public double calculateAdjustedRSquared() {
        final double n = getX().getRowDimension();
        if (isNoIntercept()) {
            return 1 - (1 - calculateRSquared()) * (n / (n - getX().getColumnDimension()));
        } else {
            return 1 - (calculateResidualSumOfSquares() * (n - 1)) /
                (calculateTotalSumOfSquares() * (n - getX().getColumnDimension()));
        }
    }

    
    @Override
    protected void newXSampleData(double[][] x) {
        super.newXSampleData(x);
        qr = new QRDecomposition(getX(), threshold);
    }

    
    @Override
    protected RealVector calculateBeta() {
        return qr.getSolver().solve(getY());
    }

    
    @Override
    protected RealMatrix calculateBetaVariance() {
        int p = getX().getColumnDimension();
        RealMatrix Raug = qr.getR().getSubMatrix(0, p - 1 , 0, p - 1);
        RealMatrix Rinv = new LUDecomposition(Raug).getSolver().getInverse();
        return Rinv.multiply(Rinv.transpose());
    }

}
