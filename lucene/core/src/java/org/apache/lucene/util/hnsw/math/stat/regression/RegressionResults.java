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

import java.io.Serializable;
import java.util.Arrays;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;


public class RegressionResults implements Serializable {

    
    private static final int SSE_IDX = 0;
    
    private static final int SST_IDX = 1;
    
    private static final int RSQ_IDX = 2;
    
    private static final int MSE_IDX = 3;
    
    private static final int ADJRSQ_IDX = 4;
    
    private static final long serialVersionUID = 1l;
    
    private final double[] parameters;
    
    private final double[][] varCovData;
    
    private final boolean isSymmetricVCD;
    
    @SuppressWarnings("unused")
    private final int rank;
    
    private final long nobs;
    
    private final boolean containsConstant;
    
    private final double[] globalFitInfo;

    
    @SuppressWarnings("unused")
    private RegressionResults() {
        this.parameters = null;
        this.varCovData = null;
        this.rank = -1;
        this.nobs = -1;
        this.containsConstant = false;
        this.isSymmetricVCD = false;
        this.globalFitInfo = null;
    }

    
    public RegressionResults(
            final double[] parameters, final double[][] varcov,
            final boolean isSymmetricCompressed,
            final long nobs, final int rank,
            final double sumy, final double sumysq, final double sse,
            final boolean containsConstant,
            final boolean copyData) {
        if (copyData) {
            this.parameters = MathArrays.copyOf(parameters);
            this.varCovData = new double[varcov.length][];
            for (int i = 0; i < varcov.length; i++) {
                this.varCovData[i] = MathArrays.copyOf(varcov[i]);
            }
        } else {
            this.parameters = parameters;
            this.varCovData = varcov;
        }
        this.isSymmetricVCD = isSymmetricCompressed;
        this.nobs = nobs;
        this.rank = rank;
        this.containsConstant = containsConstant;
        this.globalFitInfo = new double[5];
        Arrays.fill(this.globalFitInfo, Double.NaN);

        if (rank > 0) {
            this.globalFitInfo[SST_IDX] = containsConstant ?
                    (sumysq - sumy * sumy / nobs) : sumysq;
        }

        this.globalFitInfo[SSE_IDX] = sse;
        this.globalFitInfo[MSE_IDX] = this.globalFitInfo[SSE_IDX] /
                (nobs - rank);
        this.globalFitInfo[RSQ_IDX] = 1.0 -
                this.globalFitInfo[SSE_IDX] /
                this.globalFitInfo[SST_IDX];

        if (!containsConstant) {
            this.globalFitInfo[ADJRSQ_IDX] = 1.0-
                    (1.0 - this.globalFitInfo[RSQ_IDX]) *
                    ( (double) nobs / ( (double) (nobs - rank)));
        } else {
            this.globalFitInfo[ADJRSQ_IDX] = 1.0 - (sse * (nobs - 1.0)) /
                    (globalFitInfo[SST_IDX] * (nobs - rank));
        }
    }

    
    public double getParameterEstimate(int index) throws OutOfRangeException {
        if (parameters == null) {
            return Double.NaN;
        }
        if (index < 0 || index >= this.parameters.length) {
            throw new OutOfRangeException(index, 0, this.parameters.length - 1);
        }
        return this.parameters[index];
    }

    
    public double[] getParameterEstimates() {
        if (this.parameters == null) {
            return null;
        }
        return MathArrays.copyOf(parameters);
    }

    
    public double getStdErrorOfEstimate(int index) throws OutOfRangeException {
        if (parameters == null) {
            return Double.NaN;
        }
        if (index < 0 || index >= this.parameters.length) {
            throw new OutOfRangeException(index, 0, this.parameters.length - 1);
        }
        double var = this.getVcvElement(index, index);
        if (!Double.isNaN(var) && var > Double.MIN_VALUE) {
            return FastMath.sqrt(var);
        }
        return Double.NaN;
    }

    
    public double[] getStdErrorOfEstimates() {
        if (parameters == null) {
            return null;
        }
        double[] se = new double[this.parameters.length];
        for (int i = 0; i < this.parameters.length; i++) {
            double var = this.getVcvElement(i, i);
            if (!Double.isNaN(var) && var > Double.MIN_VALUE) {
                se[i] = FastMath.sqrt(var);
                continue;
            }
            se[i] = Double.NaN;
        }
        return se;
    }

    
    public double getCovarianceOfParameters(int i, int j) throws OutOfRangeException {
        if (parameters == null) {
            return Double.NaN;
        }
        if (i < 0 || i >= this.parameters.length) {
            throw new OutOfRangeException(i, 0, this.parameters.length - 1);
        }
        if (j < 0 || j >= this.parameters.length) {
            throw new OutOfRangeException(j, 0, this.parameters.length - 1);
        }
        return this.getVcvElement(i, j);
    }

    
    public int getNumberOfParameters() {
        if (this.parameters == null) {
            return -1;
        }
        return this.parameters.length;
    }

    
    public long getN() {
        return this.nobs;
    }

    
    public double getTotalSumSquares() {
        return this.globalFitInfo[SST_IDX];
    }

    
    public double getRegressionSumSquares() {
        return this.globalFitInfo[SST_IDX] - this.globalFitInfo[SSE_IDX];
    }

    
    public double getErrorSumSquares() {
        return this.globalFitInfo[ SSE_IDX];
    }

    
    public double getMeanSquareError() {
        return this.globalFitInfo[ MSE_IDX];
    }

    
    public double getRSquared() {
        return this.globalFitInfo[ RSQ_IDX];
    }

    
    public double getAdjustedRSquared() {
        return this.globalFitInfo[ ADJRSQ_IDX];
    }

    
    public boolean hasIntercept() {
        return this.containsConstant;
    }

    
    private double getVcvElement(int i, int j) {
        if (this.isSymmetricVCD) {
            if (this.varCovData.length > 1) {
                //could be stored in upper or lower triangular
                if (i == j) {
                    return varCovData[i][i];
                } else if (i >= varCovData[j].length) {
                    return varCovData[i][j];
                } else {
                    return varCovData[j][i];
                }
            } else {//could be in single array
                if (i > j) {
                    return varCovData[0][(i + 1) * i / 2 + j];
                } else {
                    return varCovData[0][(j + 1) * j / 2 + i];
                }
            }
        } else {
            return this.varCovData[i][j];
        }
    }
}
