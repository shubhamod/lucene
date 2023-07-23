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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.InsufficientDataException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.linear.NonSquareMatrixException;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Variance;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public abstract class AbstractMultipleLinearRegression implements
        MultipleLinearRegression {

    
    private RealMatrix xMatrix;

    
    private RealVector yVector;

    
    private boolean noIntercept = false;

    
    protected RealMatrix getX() {
        return xMatrix;
    }

    
    protected RealVector getY() {
        return yVector;
    }

    
    public boolean isNoIntercept() {
        return noIntercept;
    }

    
    public void setNoIntercept(boolean noIntercept) {
        this.noIntercept = noIntercept;
    }

    
    public void newSampleData(double[] data, int nobs, int nvars) {
        if (data == null) {
            throw new NullArgumentException();
        }
        if (data.length != nobs * (nvars + 1)) {
            throw new DimensionMismatchException(data.length, nobs * (nvars + 1));
        }
        if (nobs <= nvars) {
            throw new InsufficientDataException(LocalizedFormats.INSUFFICIENT_OBSERVED_POINTS_IN_SAMPLE, nobs, nvars + 1);
        }
        double[] y = new double[nobs];
        final int cols = noIntercept ? nvars: nvars + 1;
        double[][] x = new double[nobs][cols];
        int pointer = 0;
        for (int i = 0; i < nobs; i++) {
            y[i] = data[pointer++];
            if (!noIntercept) {
                x[i][0] = 1.0d;
            }
            for (int j = noIntercept ? 0 : 1; j < cols; j++) {
                x[i][j] = data[pointer++];
            }
        }
        this.xMatrix = new Array2DRowRealMatrix(x);
        this.yVector = new ArrayRealVector(y);
    }

    
    protected void newYSampleData(double[] y) {
        if (y == null) {
            throw new NullArgumentException();
        }
        if (y.length == 0) {
            throw new NoDataException();
        }
        this.yVector = new ArrayRealVector(y);
    }

    
    protected void newXSampleData(double[][] x) {
        if (x == null) {
            throw new NullArgumentException();
        }
        if (x.length == 0) {
            throw new NoDataException();
        }
        if (noIntercept) {
            this.xMatrix = new Array2DRowRealMatrix(x, true);
        } else { // Augment design matrix with initial unitary column
            final int nVars = x[0].length;
            final double[][] xAug = new double[x.length][nVars + 1];
            for (int i = 0; i < x.length; i++) {
                if (x[i].length != nVars) {
                    throw new DimensionMismatchException(x[i].length, nVars);
                }
                xAug[i][0] = 1.0d;
                System.arraycopy(x[i], 0, xAug[i], 1, nVars);
            }
            this.xMatrix = new Array2DRowRealMatrix(xAug, false);
        }
    }

    
    protected void validateSampleData(double[][] x, double[] y) throws MathIllegalArgumentException {
        if ((x == null) || (y == null)) {
            throw new NullArgumentException();
        }
        if (x.length != y.length) {
            throw new DimensionMismatchException(y.length, x.length);
        }
        if (x.length == 0) {  // Must be no y data either
            throw new NoDataException();
        }
        if (x[0].length + 1 > x.length) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS,
                    x.length, x[0].length);
        }
    }

    
    protected void validateCovarianceData(double[][] x, double[][] covariance) {
        if (x.length != covariance.length) {
            throw new DimensionMismatchException(x.length, covariance.length);
        }
        if (covariance.length > 0 && covariance.length != covariance[0].length) {
            throw new NonSquareMatrixException(covariance.length, covariance[0].length);
        }
    }

    
    public double[] estimateRegressionParameters() {
        RealVector b = calculateBeta();
        return b.toArray();
    }

    
    public double[] estimateResiduals() {
        RealVector b = calculateBeta();
        RealVector e = yVector.subtract(xMatrix.operate(b));
        return e.toArray();
    }

    
    public double[][] estimateRegressionParametersVariance() {
        return calculateBetaVariance().getData();
    }

    
    public double[] estimateRegressionParametersStandardErrors() {
        double[][] betaVariance = estimateRegressionParametersVariance();
        double sigma = calculateErrorVariance();
        int length = betaVariance[0].length;
        double[] result = new double[length];
        for (int i = 0; i < length; i++) {
            result[i] = FastMath.sqrt(sigma * betaVariance[i][i]);
        }
        return result;
    }

    
    public double estimateRegressandVariance() {
        return calculateYVariance();
    }

    
    public double estimateErrorVariance() {
        return calculateErrorVariance();

    }

    
    public double estimateRegressionStandardError() {
        return FastMath.sqrt(estimateErrorVariance());
    }

    
    protected abstract RealVector calculateBeta();

    
    protected abstract RealMatrix calculateBetaVariance();


    
    protected double calculateYVariance() {
        return new Variance().evaluate(yVector.toArray());
    }

    
    protected double calculateErrorVariance() {
        RealVector residuals = calculateResiduals();
        return residuals.dotProduct(residuals) /
               (xMatrix.getRowDimension() - xMatrix.getColumnDimension());
    }

    
    protected RealVector calculateResiduals() {
        RealVector b = calculateBeta();
        return yVector.subtract(xMatrix.operate(b));
    }

}
