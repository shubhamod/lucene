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

import org.apache.lucene.util.hnsw.math.linear.LUDecomposition;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;


public class GLSMultipleLinearRegression extends AbstractMultipleLinearRegression {

    
    private RealMatrix Omega;

    
    private RealMatrix OmegaInverse;

    
    public void newSampleData(double[] y, double[][] x, double[][] covariance) {
        validateSampleData(x, y);
        newYSampleData(y);
        newXSampleData(x);
        validateCovarianceData(x, covariance);
        newCovarianceData(covariance);
    }

    
    protected void newCovarianceData(double[][] omega){
        this.Omega = new Array2DRowRealMatrix(omega);
        this.OmegaInverse = null;
    }

    
    protected RealMatrix getOmegaInverse() {
        if (OmegaInverse == null) {
            OmegaInverse = new LUDecomposition(Omega).getSolver().getInverse();
        }
        return OmegaInverse;
    }

    
    @Override
    protected RealVector calculateBeta() {
        RealMatrix OI = getOmegaInverse();
        RealMatrix XT = getX().transpose();
        RealMatrix XTOIX = XT.multiply(OI).multiply(getX());
        RealMatrix inverse = new LUDecomposition(XTOIX).getSolver().getInverse();
        return inverse.multiply(XT).multiply(OI).operate(getY());
    }

    
    @Override
    protected RealMatrix calculateBetaVariance() {
        RealMatrix OI = getOmegaInverse();
        RealMatrix XTOIX = getX().transpose().multiply(OI).multiply(getX());
        return new LUDecomposition(XTOIX).getSolver().getInverse();
    }


    
    @Override
    protected double calculateErrorVariance() {
        RealVector residuals = calculateResiduals();
        double t = residuals.dotProduct(getOmegaInverse().operate(residuals));
        return t / (getX().getRowDimension() - getX().getColumnDimension());

    }

}
