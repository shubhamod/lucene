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

import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class SynchronizedSummaryStatistics extends SummaryStatistics {

    
    private static final long serialVersionUID = 1909861009042253704L;

    
    public SynchronizedSummaryStatistics() {
        super();
    }

    
    public SynchronizedSummaryStatistics(SynchronizedSummaryStatistics original)
    throws NullArgumentException {
        copy(original, this);
    }

    
    @Override
    public synchronized StatisticalSummary getSummary() {
        return super.getSummary();
    }

    
    @Override
    public synchronized void addValue(double value) {
        super.addValue(value);
    }

    
    @Override
    public synchronized long getN() {
        return super.getN();
    }

    
    @Override
    public synchronized double getSum() {
        return super.getSum();
    }

    
    @Override
    public synchronized double getSumsq() {
        return super.getSumsq();
    }

    
    @Override
    public synchronized double getMean() {
        return super.getMean();
    }

    
    @Override
    public synchronized double getStandardDeviation() {
        return super.getStandardDeviation();
    }

    
    @Override
    public synchronized double getQuadraticMean() {
        return super.getQuadraticMean();
    }

    
    @Override
    public synchronized double getVariance() {
        return super.getVariance();
    }

    
    @Override
    public synchronized double getPopulationVariance() {
        return super.getPopulationVariance();
    }

    
    @Override
    public synchronized double getMax() {
        return super.getMax();
    }

    
    @Override
    public synchronized double getMin() {
        return super.getMin();
    }

    
    @Override
    public synchronized double getGeometricMean() {
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
    public synchronized StorelessUnivariateStatistic getSumImpl() {
        return super.getSumImpl();
    }

    
    @Override
    public synchronized void setSumImpl(StorelessUnivariateStatistic sumImpl)
    throws MathIllegalStateException {
        super.setSumImpl(sumImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic getSumsqImpl() {
        return super.getSumsqImpl();
    }

    
    @Override
    public synchronized void setSumsqImpl(StorelessUnivariateStatistic sumsqImpl)
    throws MathIllegalStateException {
        super.setSumsqImpl(sumsqImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic getMinImpl() {
        return super.getMinImpl();
    }

    
    @Override
    public synchronized void setMinImpl(StorelessUnivariateStatistic minImpl)
    throws MathIllegalStateException {
        super.setMinImpl(minImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic getMaxImpl() {
        return super.getMaxImpl();
    }

    
    @Override
    public synchronized void setMaxImpl(StorelessUnivariateStatistic maxImpl)
    throws MathIllegalStateException {
        super.setMaxImpl(maxImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic getSumLogImpl() {
        return super.getSumLogImpl();
    }

    
    @Override
    public synchronized void setSumLogImpl(StorelessUnivariateStatistic sumLogImpl)
    throws MathIllegalStateException {
        super.setSumLogImpl(sumLogImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic getGeoMeanImpl() {
        return super.getGeoMeanImpl();
    }

    
    @Override
    public synchronized void setGeoMeanImpl(StorelessUnivariateStatistic geoMeanImpl)
    throws MathIllegalStateException {
        super.setGeoMeanImpl(geoMeanImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic getMeanImpl() {
        return super.getMeanImpl();
    }

    
    @Override
    public synchronized void setMeanImpl(StorelessUnivariateStatistic meanImpl)
    throws MathIllegalStateException {
        super.setMeanImpl(meanImpl);
    }

    
    @Override
    public synchronized StorelessUnivariateStatistic getVarianceImpl() {
        return super.getVarianceImpl();
    }

    
    @Override
    public synchronized void setVarianceImpl(StorelessUnivariateStatistic varianceImpl)
    throws MathIllegalStateException {
        super.setVarianceImpl(varianceImpl);
    }

    
    @Override
    public synchronized SynchronizedSummaryStatistics copy() {
        SynchronizedSummaryStatistics result =
            new SynchronizedSummaryStatistics();
        // No try-catch or advertised exception because arguments are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(SynchronizedSummaryStatistics source,
                            SynchronizedSummaryStatistics dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        synchronized (source) {
            synchronized (dest) {
                SummaryStatistics.copy(source, dest);
            }
        }
    }

}
