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

import org.apache.lucene.util.hnsw.math.distribution.TDistribution;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class SimpleRegression implements Serializable, UpdatingMultipleLinearRegression {

    
    private static final long serialVersionUID = -3004689053607543335L;

    
    private double sumX = 0d;

    
    private double sumXX = 0d;

    
    private double sumY = 0d;

    
    private double sumYY = 0d;

    
    private double sumXY = 0d;

    
    private long n = 0;

    
    private double xbar = 0;

    
    private double ybar = 0;

    
    private final boolean hasIntercept;
    // ---------------------Public methods--------------------------------------

    
    public SimpleRegression() {
        this(true);
    }
    
    public SimpleRegression(boolean includeIntercept) {
        super();
        hasIntercept = includeIntercept;
    }

    
    public void addData(final double x,final double y) {
        if (n == 0) {
            xbar = x;
            ybar = y;
        } else {
            if( hasIntercept ){
                final double fact1 = 1.0 + n;
                final double fact2 = n / (1.0 + n);
                final double dx = x - xbar;
                final double dy = y - ybar;
                sumXX += dx * dx * fact2;
                sumYY += dy * dy * fact2;
                sumXY += dx * dy * fact2;
                xbar += dx / fact1;
                ybar += dy / fact1;
            }
         }
        if( !hasIntercept ){
            sumXX += x * x ;
            sumYY += y * y ;
            sumXY += x * y ;
        }
        sumX += x;
        sumY += y;
        n++;
    }

    
    public void append(SimpleRegression reg) {
        if (n == 0) {
            xbar = reg.xbar;
            ybar = reg.ybar;
            sumXX = reg.sumXX;
            sumYY = reg.sumYY;
            sumXY = reg.sumXY;
        } else {
            if (hasIntercept) {
                final double fact1 = reg.n / (double) (reg.n + n);
                final double fact2 = n * reg.n / (double) (reg.n + n);
                final double dx = reg.xbar - xbar;
                final double dy = reg.ybar - ybar;
                sumXX += reg.sumXX + dx * dx * fact2;
                sumYY += reg.sumYY + dy * dy * fact2;
                sumXY += reg.sumXY + dx * dy * fact2;
                xbar += dx * fact1;
                ybar += dy * fact1;
            }else{
                sumXX += reg.sumXX;
                sumYY += reg.sumYY;
                sumXY += reg.sumXY;
            }
        }
        sumX += reg.sumX;
        sumY += reg.sumY;
        n += reg.n;
    }

    
    public void removeData(final double x,final double y) {
        if (n > 0) {
            if (hasIntercept) {
                final double fact1 = n - 1.0;
                final double fact2 = n / (n - 1.0);
                final double dx = x - xbar;
                final double dy = y - ybar;
                sumXX -= dx * dx * fact2;
                sumYY -= dy * dy * fact2;
                sumXY -= dx * dy * fact2;
                xbar -= dx / fact1;
                ybar -= dy / fact1;
            } else {
                final double fact1 = n - 1.0;
                sumXX -= x * x;
                sumYY -= y * y;
                sumXY -= x * y;
                xbar -= x / fact1;
                ybar -= y / fact1;
            }
             sumX -= x;
             sumY -= y;
             n--;
        }
    }

    
    public void addData(final double[][] data) throws ModelSpecificationException {
        for (int i = 0; i < data.length; i++) {
            if( data[i].length < 2 ){
               throw new ModelSpecificationException(LocalizedFormats.INVALID_REGRESSION_OBSERVATION,
                    data[i].length, 2);
            }
            addData(data[i][0], data[i][1]);
        }
    }

    
    public void addObservation(final double[] x,final double y)
    throws ModelSpecificationException {
        if( x == null || x.length == 0 ){
            throw new ModelSpecificationException(LocalizedFormats.INVALID_REGRESSION_OBSERVATION,x!=null?x.length:0, 1);
        }
        addData( x[0], y );
    }

    
    public void addObservations(final double[][] x,final double[] y) throws ModelSpecificationException {
        if ((x == null) || (y == null) || (x.length != y.length)) {
            throw new ModelSpecificationException(
                  LocalizedFormats.DIMENSIONS_MISMATCH_SIMPLE,
                  (x == null) ? 0 : x.length,
                  (y == null) ? 0 : y.length);
        }
        boolean obsOk=true;
        for( int i = 0 ; i < x.length; i++){
            if( x[i] == null || x[i].length == 0 ){
                obsOk = false;
            }
        }
        if( !obsOk ){
            throw new ModelSpecificationException(
                  LocalizedFormats.NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS,
                  0, 1);
        }
        for( int i = 0 ; i < x.length ; i++){
            addData( x[i][0], y[i] );
        }
    }

    
    public void removeData(double[][] data) {
        for (int i = 0; i < data.length && n > 0; i++) {
            removeData(data[i][0], data[i][1]);
        }
    }

    
    public void clear() {
        sumX = 0d;
        sumXX = 0d;
        sumY = 0d;
        sumYY = 0d;
        sumXY = 0d;
        n = 0;
    }

    
    public long getN() {
        return n;
    }

    
    public double predict(final double x) {
        final double b1 = getSlope();
        if (hasIntercept) {
            return getIntercept(b1) + b1 * x;
        }
        return b1 * x;
    }

    
    public double getIntercept() {
        return hasIntercept ? getIntercept(getSlope()) : 0.0;
    }

    
    public boolean hasIntercept() {
        return hasIntercept;
    }

    
    public double getSlope() {
        if (n < 2) {
            return Double.NaN; //not enough data
        }
        if (FastMath.abs(sumXX) < 10 * Double.MIN_VALUE) {
            return Double.NaN; //not enough variation in x
        }
        return sumXY / sumXX;
    }

    
    public double getSumSquaredErrors() {
        return FastMath.max(0d, sumYY - sumXY * sumXY / sumXX);
    }

    
    public double getTotalSumSquares() {
        if (n < 2) {
            return Double.NaN;
        }
        return sumYY;
    }

    
    public double getXSumSquares() {
        if (n < 2) {
            return Double.NaN;
        }
        return sumXX;
    }

    
    public double getSumOfCrossProducts() {
        return sumXY;
    }

    
    public double getRegressionSumSquares() {
        return getRegressionSumSquares(getSlope());
    }

    
    public double getMeanSquareError() {
        if (n < 3) {
            return Double.NaN;
        }
        return hasIntercept ? (getSumSquaredErrors() / (n - 2)) : (getSumSquaredErrors() / (n - 1));
    }

    
    public double getR() {
        double b1 = getSlope();
        double result = FastMath.sqrt(getRSquare());
        if (b1 < 0) {
            result = -result;
        }
        return result;
    }

    
    public double getRSquare() {
        double ssto = getTotalSumSquares();
        return (ssto - getSumSquaredErrors()) / ssto;
    }

    
    public double getInterceptStdErr() {
        if( !hasIntercept ){
            return Double.NaN;
        }
        return FastMath.sqrt(
            getMeanSquareError() * ((1d / n) + (xbar * xbar) / sumXX));
    }

    
    public double getSlopeStdErr() {
        return FastMath.sqrt(getMeanSquareError() / sumXX);
    }

    
    public double getSlopeConfidenceInterval() throws OutOfRangeException {
        return getSlopeConfidenceInterval(0.05d);
    }

    
    public double getSlopeConfidenceInterval(final double alpha)
    throws OutOfRangeException {
        if (n < 3) {
            return Double.NaN;
        }
        if (alpha >= 1 || alpha <= 0) {
            throw new OutOfRangeException(LocalizedFormats.SIGNIFICANCE_LEVEL,
                                          alpha, 0, 1);
        }
        // No advertised NotStrictlyPositiveException here - will return NaN above
        TDistribution distribution = new TDistribution(n - 2);
        return getSlopeStdErr() *
            distribution.inverseCumulativeProbability(1d - alpha / 2d);
    }

    
    public double getSignificance() {
        if (n < 3) {
            return Double.NaN;
        }
        // No advertised NotStrictlyPositiveException here - will return NaN above
        TDistribution distribution = new TDistribution(n - 2);
        return 2d * (1.0 - distribution.cumulativeProbability(
                    FastMath.abs(getSlope()) / getSlopeStdErr()));
    }

    // ---------------------Private methods-----------------------------------

    
    private double getIntercept(final double slope) {
      if( hasIntercept){
        return (sumY - slope * sumX) / n;
      }
      return 0.0;
    }

    
    private double getRegressionSumSquares(final double slope) {
        return slope * slope * sumXX;
    }

    
    public RegressionResults regress() throws ModelSpecificationException, NoDataException {
        if (hasIntercept) {
            if (n < 3) {
                throw new NoDataException(LocalizedFormats.NOT_ENOUGH_DATA_REGRESSION);
            }
            if (FastMath.abs(sumXX) > Precision.SAFE_MIN) {
                final double[] params = new double[] { getIntercept(), getSlope() };
                final double mse = getMeanSquareError();
                final double _syy = sumYY + sumY * sumY / n;
                final double[] vcv = new double[] { mse * (xbar * xbar / sumXX + 1.0 / n), -xbar * mse / sumXX, mse / sumXX };
                return new RegressionResults(params, new double[][] { vcv }, true, n, 2, sumY, _syy, getSumSquaredErrors(), true,
                        false);
            } else {
                final double[] params = new double[] { sumY / n, Double.NaN };
                // final double mse = getMeanSquareError();
                final double[] vcv = new double[] { ybar / (n - 1.0), Double.NaN, Double.NaN };
                return new RegressionResults(params, new double[][] { vcv }, true, n, 1, sumY, sumYY, getSumSquaredErrors(), true,
                        false);
            }
        } else {
            if (n < 2) {
                throw new NoDataException(LocalizedFormats.NOT_ENOUGH_DATA_REGRESSION);
            }
            if (!Double.isNaN(sumXX)) {
                final double[] vcv = new double[] { getMeanSquareError() / sumXX };
                final double[] params = new double[] { sumXY / sumXX };
                return new RegressionResults(params, new double[][] { vcv }, true, n, 1, sumY, sumYY, getSumSquaredErrors(), false,
                        false);
            } else {
                final double[] vcv = new double[] { Double.NaN };
                final double[] params = new double[] { Double.NaN };
                return new RegressionResults(params, new double[][] { vcv }, true, n, 1, Double.NaN, Double.NaN, Double.NaN, false,
                        false);
            }
        }
    }

    
    public RegressionResults regress(int[] variablesToInclude) throws MathIllegalArgumentException{
        if( variablesToInclude == null || variablesToInclude.length == 0){
          throw new MathIllegalArgumentException(LocalizedFormats.ARRAY_ZERO_LENGTH_OR_NULL_NOT_ALLOWED);
        }
        if( variablesToInclude.length > 2 || (variablesToInclude.length > 1 && !hasIntercept) ){
            throw new ModelSpecificationException(
                    LocalizedFormats.ARRAY_SIZE_EXCEEDS_MAX_VARIABLES,
                    (variablesToInclude.length > 1 && !hasIntercept) ? 1 : 2);
        }

        if( hasIntercept ){
            if( variablesToInclude.length == 2 ){
                if( variablesToInclude[0] == 1 ){
                    throw new ModelSpecificationException(LocalizedFormats.NOT_INCREASING_SEQUENCE);
                }else if( variablesToInclude[0] != 0 ){
                    throw new OutOfRangeException( variablesToInclude[0], 0,1 );
                }
                if( variablesToInclude[1] != 1){
                     throw new OutOfRangeException( variablesToInclude[0], 0,1 );
                }
                return regress();
            }else{
                if( variablesToInclude[0] != 1 && variablesToInclude[0] != 0 ){
                     throw new OutOfRangeException( variablesToInclude[0],0,1 );
                }
                final double _mean = sumY * sumY / n;
                final double _syy = sumYY + _mean;
                if( variablesToInclude[0] == 0 ){
                    //just the mean
                    final double[] vcv = new double[]{ sumYY/(((n-1)*n)) };
                    final double[] params = new double[]{ ybar };
                    return new RegressionResults(
                      params, new double[][]{vcv}, true, n, 1,
                      sumY, _syy+_mean, sumYY,true,false);

                }else if( variablesToInclude[0] == 1){
                    //final double _syy = sumYY + sumY * sumY / ((double) n);
                    final double _sxx = sumXX + sumX * sumX / n;
                    final double _sxy = sumXY + sumX * sumY / n;
                    final double _sse = FastMath.max(0d, _syy - _sxy * _sxy / _sxx);
                    final double _mse = _sse/((n-1));
                    if( !Double.isNaN(_sxx) ){
                        final double[] vcv = new double[]{ _mse / _sxx };
                        final double[] params = new double[]{ _sxy/_sxx };
                        return new RegressionResults(
                                    params, new double[][]{vcv}, true, n, 1,
                                    sumY, _syy, _sse,false,false);
                    }else{
                        final double[] vcv = new double[]{Double.NaN };
                        final double[] params = new double[]{ Double.NaN };
                        return new RegressionResults(
                                    params, new double[][]{vcv}, true, n, 1,
                                    Double.NaN, Double.NaN, Double.NaN,false,false);
                    }
                }
            }
        }else{
            if( variablesToInclude[0] != 0 ){
                throw new OutOfRangeException(variablesToInclude[0],0,0);
            }
            return regress();
        }

        return null;
    }
}
