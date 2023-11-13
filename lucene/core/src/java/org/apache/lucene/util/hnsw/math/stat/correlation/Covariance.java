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
package org.apache.lucene.util.hnsw.math.stat.correlation;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.BlockRealMatrix;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Mean;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Variance;


public class Covariance {

    
    private final RealMatrix covarianceMatrix;

    
    
    private final int n;

    
    public Covariance() {
        super();
        covarianceMatrix = null;
        n = 0;
    }

    
    public Covariance(double[][] data, boolean biasCorrected)
    throws MathIllegalArgumentException, NotStrictlyPositiveException {
        this(new BlockRealMatrix(data), biasCorrected);
    }

    
    public Covariance(double[][] data)
    throws MathIllegalArgumentException, NotStrictlyPositiveException {
        this(data, true);
    }

    
    public Covariance(RealMatrix matrix, boolean biasCorrected)
    throws MathIllegalArgumentException {
       checkSufficientData(matrix);
       n = matrix.getRowDimension();
       covarianceMatrix = computeCovarianceMatrix(matrix, biasCorrected);
    }

    
    public Covariance(RealMatrix matrix) throws MathIllegalArgumentException {
        this(matrix, true);
    }

    
    public RealMatrix getCovarianceMatrix() {
        return covarianceMatrix;
    }

    
    public int getN() {
        return n;
    }

    
    protected RealMatrix computeCovarianceMatrix(RealMatrix matrix, boolean biasCorrected)
    throws MathIllegalArgumentException {
        int dimension = matrix.getColumnDimension();
        Variance variance = new Variance(biasCorrected);
        RealMatrix outMatrix = new BlockRealMatrix(dimension, dimension);
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < i; j++) {
              double cov = covariance(matrix.getColumn(i), matrix.getColumn(j), biasCorrected);
              outMatrix.setEntry(i, j, cov);
              outMatrix.setEntry(j, i, cov);
            }
            outMatrix.setEntry(i, i, variance.evaluate(matrix.getColumn(i)));
        }
        return outMatrix;
    }

    
    protected RealMatrix computeCovarianceMatrix(RealMatrix matrix)
    throws MathIllegalArgumentException {
        return computeCovarianceMatrix(matrix, true);
    }

    
    protected RealMatrix computeCovarianceMatrix(double[][] data, boolean biasCorrected)
    throws MathIllegalArgumentException, NotStrictlyPositiveException {
        return computeCovarianceMatrix(new BlockRealMatrix(data), biasCorrected);
    }

    
    protected RealMatrix computeCovarianceMatrix(double[][] data)
    throws MathIllegalArgumentException, NotStrictlyPositiveException {
        return computeCovarianceMatrix(data, true);
    }

    
    public double covariance(final double[] xArray, final double[] yArray, boolean biasCorrected)
        throws MathIllegalArgumentException {
        Mean mean = new Mean();
        double result = 0d;
        int length = xArray.length;
        if (length != yArray.length) {
            throw new MathIllegalArgumentException(
                  LocalizedFormats.DIMENSIONS_MISMATCH_SIMPLE, length, yArray.length);
        } else if (length < 2) {
            throw new MathIllegalArgumentException(
                  LocalizedFormats.INSUFFICIENT_OBSERVED_POINTS_IN_SAMPLE, length, 2);
        } else {
            double xMean = mean.evaluate(xArray);
            double yMean = mean.evaluate(yArray);
            for (int i = 0; i < length; i++) {
                double xDev = xArray[i] - xMean;
                double yDev = yArray[i] - yMean;
                result += (xDev * yDev - result) / (i + 1);
            }
        }
        return biasCorrected ? result * ((double) length / (double)(length - 1)) : result;
    }

    
    public double covariance(final double[] xArray, final double[] yArray)
        throws MathIllegalArgumentException {
        return covariance(xArray, yArray, true);
    }

    
    private void checkSufficientData(final RealMatrix matrix) throws MathIllegalArgumentException {
        int nRows = matrix.getRowDimension();
        int nCols = matrix.getColumnDimension();
        if (nRows < 2 || nCols < 1) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.INSUFFICIENT_ROWS_AND_COLUMNS,
                    nRows, nCols);
        }
    }
}
