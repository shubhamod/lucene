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
package org.apache.lucene.util.hnsw.math.stat.descriptive;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.GeometricMean;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Mean;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.VectorialCovariance;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Max;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Min;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.Sum;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.SumOfLogs;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.SumOfSquares;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.Precision;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class MultivariateSummaryStatistics
    implements StatisticalMultivariateSummary, Serializable {

    
    private static final long serialVersionUID = 2271900808994826718L;

    
    private int k;

    
    private long n = 0;

    
    private StorelessUnivariateStatistic[] sumImpl;

    
    private StorelessUnivariateStatistic[] sumSqImpl;

    
    private StorelessUnivariateStatistic[] minImpl;

    
    private StorelessUnivariateStatistic[] maxImpl;

    
    private StorelessUnivariateStatistic[] sumLogImpl;

    
    private StorelessUnivariateStatistic[] geoMeanImpl;

    
    private StorelessUnivariateStatistic[] meanImpl;

    
    private VectorialCovariance covarianceImpl;

    
    public MultivariateSummaryStatistics(int k, boolean isCovarianceBiasCorrected) {
        this.k = k;

        sumImpl     = new StorelessUnivariateStatistic[k];
        sumSqImpl   = new StorelessUnivariateStatistic[k];
        minImpl     = new StorelessUnivariateStatistic[k];
        maxImpl     = new StorelessUnivariateStatistic[k];
        sumLogImpl  = new StorelessUnivariateStatistic[k];
        geoMeanImpl = new StorelessUnivariateStatistic[k];
        meanImpl    = new StorelessUnivariateStatistic[k];

        for (int i = 0; i < k; ++i) {
            sumImpl[i]     = new Sum();
            sumSqImpl[i]   = new SumOfSquares();
            minImpl[i]     = new Min();
            maxImpl[i]     = new Max();
            sumLogImpl[i]  = new SumOfLogs();
            geoMeanImpl[i] = new GeometricMean();
            meanImpl[i]    = new Mean();
        }

        covarianceImpl =
            new VectorialCovariance(k, isCovarianceBiasCorrected);

    }

    
    public void addValue(double[] value) throws DimensionMismatchException {
        checkDimension(value.length);
        for (int i = 0; i < k; ++i) {
            double v = value[i];
            sumImpl[i].increment(v);
            sumSqImpl[i].increment(v);
            minImpl[i].increment(v);
            maxImpl[i].increment(v);
            sumLogImpl[i].increment(v);
            geoMeanImpl[i].increment(v);
            meanImpl[i].increment(v);
        }
        covarianceImpl.increment(value);
        n++;
    }

    
    public int getDimension() {
        return k;
    }

    
    public long getN() {
        return n;
    }

    
    private double[] getResults(StorelessUnivariateStatistic[] stats) {
        double[] results = new double[stats.length];
        for (int i = 0; i < results.length; ++i) {
            results[i] = stats[i].getResult();
        }
        return results;
    }

    
    public double[] getSum() {
        return getResults(sumImpl);
    }

    
    public double[] getSumSq() {
        return getResults(sumSqImpl);
    }

    
    public double[] getSumLog() {
        return getResults(sumLogImpl);
    }

    
    public double[] getMean() {
        return getResults(meanImpl);
    }

    
    public double[] getStandardDeviation() {
        double[] stdDev = new double[k];
        if (getN() < 1) {
            Arrays.fill(stdDev, Double.NaN);
        } else if (getN() < 2) {
            Arrays.fill(stdDev, 0.0);
        } else {
            RealMatrix matrix = covarianceImpl.getResult();
            for (int i = 0; i < k; ++i) {
                stdDev[i] = FastMath.sqrt(matrix.getEntry(i, i));
            }
        }
        return stdDev;
    }

    
    public RealMatrix getCovariance() {
        return covarianceImpl.getResult();
    }

    
    public double[] getMax() {
        return getResults(maxImpl);
    }

    
    public double[] getMin() {
        return getResults(minImpl);
    }

    
    public double[] getGeometricMean() {
        return getResults(geoMeanImpl);
    }

    
    @Override
    public String toString() {
        final String separator = ", ";
        final String suffix = System.getProperty("line.separator");
        StringBuilder outBuffer = new StringBuilder();
        outBuffer.append("MultivariateSummaryStatistics:" + suffix);
        outBuffer.append("n: " + getN() + suffix);
        append(outBuffer, getMin(), "min: ", separator, suffix);
        append(outBuffer, getMax(), "max: ", separator, suffix);
        append(outBuffer, getMean(), "mean: ", separator, suffix);
        append(outBuffer, getGeometricMean(), "geometric mean: ", separator, suffix);
        append(outBuffer, getSumSq(), "sum of squares: ", separator, suffix);
        append(outBuffer, getSumLog(), "sum of logarithms: ", separator, suffix);
        append(outBuffer, getStandardDeviation(), "standard deviation: ", separator, suffix);
        outBuffer.append("covariance: " + getCovariance().toString() + suffix);
        return outBuffer.toString();
    }

    
    private void append(StringBuilder buffer, double[] data,
                        String prefix, String separator, String suffix) {
        buffer.append(prefix);
        for (int i = 0; i < data.length; ++i) {
            if (i > 0) {
                buffer.append(separator);
            }
            buffer.append(data[i]);
        }
        buffer.append(suffix);
    }

    
    public void clear() {
        this.n = 0;
        for (int i = 0; i < k; ++i) {
            minImpl[i].clear();
            maxImpl[i].clear();
            sumImpl[i].clear();
            sumLogImpl[i].clear();
            sumSqImpl[i].clear();
            geoMeanImpl[i].clear();
            meanImpl[i].clear();
        }
        covarianceImpl.clear();
    }

    
    @Override
    public boolean equals(Object object) {
        if (object == this ) {
            return true;
        }
        if (object instanceof MultivariateSummaryStatistics == false) {
            return false;
        }
        MultivariateSummaryStatistics stat = (MultivariateSummaryStatistics) object;
        return MathArrays.equalsIncludingNaN(stat.getGeometricMean(), getGeometricMean()) &&
               MathArrays.equalsIncludingNaN(stat.getMax(),           getMax())           &&
               MathArrays.equalsIncludingNaN(stat.getMean(),          getMean())          &&
               MathArrays.equalsIncludingNaN(stat.getMin(),           getMin())           &&
               Precision.equalsIncludingNaN(stat.getN(),             getN())             &&
               MathArrays.equalsIncludingNaN(stat.getSum(),           getSum())           &&
               MathArrays.equalsIncludingNaN(stat.getSumSq(),         getSumSq())         &&
               MathArrays.equalsIncludingNaN(stat.getSumLog(),        getSumLog())        &&
               stat.getCovariance().equals( getCovariance());
    }

    
    @Override
    public int hashCode() {
        int result = 31 + MathUtils.hash(getGeometricMean());
        result = result * 31 + MathUtils.hash(getGeometricMean());
        result = result * 31 + MathUtils.hash(getMax());
        result = result * 31 + MathUtils.hash(getMean());
        result = result * 31 + MathUtils.hash(getMin());
        result = result * 31 + MathUtils.hash(getN());
        result = result * 31 + MathUtils.hash(getSum());
        result = result * 31 + MathUtils.hash(getSumSq());
        result = result * 31 + MathUtils.hash(getSumLog());
        result = result * 31 + getCovariance().hashCode();
        return result;
    }

    // Getters and setters for statistics implementations
    
    private void setImpl(StorelessUnivariateStatistic[] newImpl,
                         StorelessUnivariateStatistic[] oldImpl) throws MathIllegalStateException,
                         DimensionMismatchException {
        checkEmpty();
        checkDimension(newImpl.length);
        System.arraycopy(newImpl, 0, oldImpl, 0, newImpl.length);
    }

    
    public StorelessUnivariateStatistic[] getSumImpl() {
        return sumImpl.clone();
    }

    
    public void setSumImpl(StorelessUnivariateStatistic[] sumImpl)
    throws MathIllegalStateException, DimensionMismatchException {
        setImpl(sumImpl, this.sumImpl);
    }

    
    public StorelessUnivariateStatistic[] getSumsqImpl() {
        return sumSqImpl.clone();
    }

    
    public void setSumsqImpl(StorelessUnivariateStatistic[] sumsqImpl)
    throws MathIllegalStateException, DimensionMismatchException {
        setImpl(sumsqImpl, this.sumSqImpl);
    }

    
    public StorelessUnivariateStatistic[] getMinImpl() {
        return minImpl.clone();
    }

    
    public void setMinImpl(StorelessUnivariateStatistic[] minImpl)
    throws MathIllegalStateException, DimensionMismatchException {
        setImpl(minImpl, this.minImpl);
    }

    
    public StorelessUnivariateStatistic[] getMaxImpl() {
        return maxImpl.clone();
    }

    
    public void setMaxImpl(StorelessUnivariateStatistic[] maxImpl)
    throws MathIllegalStateException, DimensionMismatchException{
        setImpl(maxImpl, this.maxImpl);
    }

    
    public StorelessUnivariateStatistic[] getSumLogImpl() {
        return sumLogImpl.clone();
    }

    
    public void setSumLogImpl(StorelessUnivariateStatistic[] sumLogImpl)
    throws MathIllegalStateException, DimensionMismatchException{
        setImpl(sumLogImpl, this.sumLogImpl);
    }

    
    public StorelessUnivariateStatistic[] getGeoMeanImpl() {
        return geoMeanImpl.clone();
    }

    
    public void setGeoMeanImpl(StorelessUnivariateStatistic[] geoMeanImpl)
    throws MathIllegalStateException, DimensionMismatchException {
        setImpl(geoMeanImpl, this.geoMeanImpl);
    }

    
    public StorelessUnivariateStatistic[] getMeanImpl() {
        return meanImpl.clone();
    }

    
    public void setMeanImpl(StorelessUnivariateStatistic[] meanImpl)
    throws MathIllegalStateException, DimensionMismatchException{
        setImpl(meanImpl, this.meanImpl);
    }

    
    private void checkEmpty() throws MathIllegalStateException {
        if (n > 0) {
            throw new MathIllegalStateException(
                    LocalizedFormats.VALUES_ADDED_BEFORE_CONFIGURING_STATISTIC, n);
        }
    }

    
    private void checkDimension(int dimension) throws DimensionMismatchException {
        if (dimension != k) {
            throw new DimensionMismatchException(dimension, k);
        }
    }
}
