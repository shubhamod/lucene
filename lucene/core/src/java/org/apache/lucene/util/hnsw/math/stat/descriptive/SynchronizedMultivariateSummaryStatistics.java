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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;


public class SynchronizedMultivariateSummaryStatistics
    extends MultivariateSummaryStatistics {

    
    private static final long serialVersionUID = 7099834153347155363L;

    
    public SynchronizedMultivariateSummaryStatistics(int k, boolean isCovarianceBiasCorrected) {
        super(k, isCovarianceBiasCorrected);
    }

    
    @Override
    public synchronized void addValue(double[] value) throws DimensionMismatchException {
      super.addValue(value);
    }

    
    @Override
    public synchronized int getDimension() {
        return super.getDimension();
    }

    
    @Override
    public synchronized long getN() {
        return super.getN();
    }

    
    @Override
    public synchronized double[] getSum() {
        return super.getSum();
    }

    
    @Override
    public synchronized double[] getSumSq() {
        return super.getSumSq();
    }

    
    @Override
    public synchronized double[] getSumLog() {
        return super.getSumLog();
    }

    
    @Override
    public synchronized double[] getMean() {
        return super.getMean();
    }

    
    @Override
    public synchronized double[] getStandardDeviation() {
        return super.getStandardDeviation();
    }

    
    @Override
    public synchronized RealMatrix getCovariance() {
        return super.getCovariance();
    }

    
    @Override
    public synchronized double[] getMax() {
        return super.getMax();
    }

    
    @Override
    public synchronized double[] getMin() {
        return super.getMin();
    }

    
    @Override
    public synchronized double[] getGeometricMean() {
        return super.getGeometricMean();
    }

    
    @Override
    public synchronized String toString() {
        return super.toString();
    }

    
    @Override
    public synchronized void clear() {
        super.clear();
    }

    
    @Override
    public synchronized boolean equals(Object object) {
        return super.equals(object);
    }

    
    @Override
    public synchronized int hashCode() {
        return super.hashCode();
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic[] getSumImpl() {
        return super.getSumImpl();
    }

    
    @Override
    public synchronized void setSumImpl(StorelessUnivariateStatistic[] sumImpl)
    throws DimensionMismatchException, MathIllegalStateException {
        super.setSumImpl(sumImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic[] getSumsqImpl() {
        return super.getSumsqImpl();
    }

    
    @Override
    public synchronized void setSumsqImpl(StorelessUnivariateStatistic[] sumsqImpl)
    throws DimensionMismatchException, MathIllegalStateException {
        super.setSumsqImpl(sumsqImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic[] getMinImpl() {
        return super.getMinImpl();
    }

    
    @Override
    public synchronized void setMinImpl(StorelessUnivariateStatistic[] minImpl)
    throws DimensionMismatchException, MathIllegalStateException {
        super.setMinImpl(minImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic[] getMaxImpl() {
        return super.getMaxImpl();
    }

    
    @Override
    public synchronized void setMaxImpl(StorelessUnivariateStatistic[] maxImpl)
    throws DimensionMismatchException, MathIllegalStateException{
        super.setMaxImpl(maxImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic[] getSumLogImpl() {
        return super.getSumLogImpl();
    }

    
    @Override
    public synchronized void setSumLogImpl(StorelessUnivariateStatistic[] sumLogImpl)
    throws DimensionMismatchException, MathIllegalStateException {
        super.setSumLogImpl(sumLogImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic[] getGeoMeanImpl() {
        return super.getGeoMeanImpl();
    }

    
    @Override
    public synchronized void setGeoMeanImpl(StorelessUnivariateStatistic[] geoMeanImpl)
    throws DimensionMismatchException, MathIllegalStateException {
        super.setGeoMeanImpl(geoMeanImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic[] getMeanImpl() {
        return super.getMeanImpl();
    }

    
    @Override
    public synchronized void setMeanImpl(StorelessUnivariateStatistic[] meanImpl)
    throws DimensionMismatchException, MathIllegalStateException {
        super.setMeanImpl(meanImpl);
    }
}
