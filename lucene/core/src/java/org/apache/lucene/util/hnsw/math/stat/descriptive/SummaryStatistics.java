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

import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.GeometricMean;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Mean;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.SecondMoment;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Variance;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Max;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Min;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.Sum;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.SumOfLogs;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.SumOfSquares;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.Precision;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class SummaryStatistics implements StatisticalSummary, Serializable {

    
    private static final long serialVersionUID = -2021321786743555871L;

    
    private long n = 0;

    
    private SecondMoment secondMoment = new SecondMoment();

    
    private Sum sum = new Sum();

    
    private SumOfSquares sumsq = new SumOfSquares();

    
    private Min min = new Min();

    
    private Max max = new Max();

    
    private SumOfLogs sumLog = new SumOfLogs();

    
    private GeometricMean geoMean = new GeometricMean(sumLog);

    
    private Mean mean = new Mean(secondMoment);

    
    private Variance variance = new Variance(secondMoment);

    
    private StorelessUnivariateStatistic sumImpl = sum;

    
    private StorelessUnivariateStatistic sumsqImpl = sumsq;

    
    private StorelessUnivariateStatistic minImpl = min;

    
    private StorelessUnivariateStatistic maxImpl = max;

    
    private StorelessUnivariateStatistic sumLogImpl = sumLog;

    
    private StorelessUnivariateStatistic geoMeanImpl = geoMean;

    
    private StorelessUnivariateStatistic meanImpl = mean;

    
    private StorelessUnivariateStatistic varianceImpl = variance;

    
    public SummaryStatistics() {
    }

    
    public SummaryStatistics(SummaryStatistics original) throws NullArgumentException {
        copy(original, this);
    }

    
    public StatisticalSummary getSummary() {
        return new StatisticalSummaryValues(getMean(), getVariance(), getN(),
                getMax(), getMin(), getSum());
    }

    
    public void addValue(double value) {
        sumImpl.increment(value);
        sumsqImpl.increment(value);
        minImpl.increment(value);
        maxImpl.increment(value);
        sumLogImpl.increment(value);
        secondMoment.increment(value);
        // If mean, variance or geomean have been overridden,
        // need to increment these
        if (meanImpl != mean) {
            meanImpl.increment(value);
        }
        if (varianceImpl != variance) {
            varianceImpl.increment(value);
        }
        if (geoMeanImpl != geoMean) {
            geoMeanImpl.increment(value);
        }
        n++;
    }

    
    public long getN() {
        return n;
    }

    
    public double getSum() {
        return sumImpl.getResult();
    }

    
    public double getSumsq() {
        return sumsqImpl.getResult();
    }

    
    public double getMean() {
        return meanImpl.getResult();
    }

    
    public double getStandardDeviation() {
        double stdDev = Double.NaN;
        if (getN() > 0) {
            if (getN() > 1) {
                stdDev = FastMath.sqrt(getVariance());
            } else {
                stdDev = 0.0;
            }
        }
        return stdDev;
    }

    
    public double getQuadraticMean() {
        final long size = getN();
        return size > 0 ? FastMath.sqrt(getSumsq() / size) : Double.NaN;
    }

    
    public double getVariance() {
        return varianceImpl.getResult();
    }

    
    public double getPopulationVariance() {
        Variance populationVariance = new Variance(secondMoment);
        populationVariance.setBiasCorrected(false);
        return populationVariance.getResult();
    }

    
    public double getMax() {
        return maxImpl.getResult();
    }

    
    public double getMin() {
        return minImpl.getResult();
    }

    
    public double getGeometricMean() {
        return geoMeanImpl.getResult();
    }

    
    public double getSumOfLogs() {
        return sumLogImpl.getResult();
    }

    
    public double getSecondMoment() {
        return secondMoment.getResult();
    }

    
    @Override
    public String toString() {
        StringBuilder outBuffer = new StringBuilder();
        String endl = "\n";
        outBuffer.append("SummaryStatistics:").append(endl);
        outBuffer.append("n: ").append(getN()).append(endl);
        outBuffer.append("min: ").append(getMin()).append(endl);
        outBuffer.append("max: ").append(getMax()).append(endl);
        outBuffer.append("sum: ").append(getSum()).append(endl);
        outBuffer.append("mean: ").append(getMean()).append(endl);
        outBuffer.append("geometric mean: ").append(getGeometricMean())
            .append(endl);
        outBuffer.append("variance: ").append(getVariance()).append(endl);
        outBuffer.append("population variance: ").append(getPopulationVariance()).append(endl);
        outBuffer.append("second moment: ").append(getSecondMoment()).append(endl);
        outBuffer.append("sum of squares: ").append(getSumsq()).append(endl);
        outBuffer.append("standard deviation: ").append(getStandardDeviation())
            .append(endl);
        outBuffer.append("sum of logs: ").append(getSumOfLogs()).append(endl);
        return outBuffer.toString();
    }

    
    public void clear() {
        this.n = 0;
        minImpl.clear();
        maxImpl.clear();
        sumImpl.clear();
        sumLogImpl.clear();
        sumsqImpl.clear();
        geoMeanImpl.clear();
        secondMoment.clear();
        if (meanImpl != mean) {
            meanImpl.clear();
        }
        if (varianceImpl != variance) {
            varianceImpl.clear();
        }
    }

    
    @Override
    public boolean equals(Object object) {
        if (object == this) {
            return true;
        }
        if (object instanceof SummaryStatistics == false) {
            return false;
        }
        SummaryStatistics stat = (SummaryStatistics)object;
        return Precision.equalsIncludingNaN(stat.getGeometricMean(), getGeometricMean()) &&
               Precision.equalsIncludingNaN(stat.getMax(),           getMax())           &&
               Precision.equalsIncludingNaN(stat.getMean(),          getMean())          &&
               Precision.equalsIncludingNaN(stat.getMin(),           getMin())           &&
               Precision.equalsIncludingNaN(stat.getN(),             getN())             &&
               Precision.equalsIncludingNaN(stat.getSum(),           getSum())           &&
               Precision.equalsIncludingNaN(stat.getSumsq(),         getSumsq())         &&
               Precision.equalsIncludingNaN(stat.getVariance(),      getVariance());
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
        result = result * 31 + MathUtils.hash(getSumsq());
        result = result * 31 + MathUtils.hash(getVariance());
        return result;
    }

    // Getters and setters for statistics implementations
    
    public StorelessUnivariateStatistic getSumImpl() {
        return sumImpl;
    }

    
    public void setSumImpl(StorelessUnivariateStatistic sumImpl)
    throws MathIllegalStateException {
        checkEmpty();
        this.sumImpl = sumImpl;
    }

    
    public StorelessUnivariateStatistic getSumsqImpl() {
        return sumsqImpl;
    }

    
    public void setSumsqImpl(StorelessUnivariateStatistic sumsqImpl)
    throws MathIllegalStateException {
        checkEmpty();
        this.sumsqImpl = sumsqImpl;
    }

    
    public StorelessUnivariateStatistic getMinImpl() {
        return minImpl;
    }

    
    public void setMinImpl(StorelessUnivariateStatistic minImpl)
    throws MathIllegalStateException {
        checkEmpty();
        this.minImpl = minImpl;
    }

    
    public StorelessUnivariateStatistic getMaxImpl() {
        return maxImpl;
    }

    
    public void setMaxImpl(StorelessUnivariateStatistic maxImpl)
    throws MathIllegalStateException {
        checkEmpty();
        this.maxImpl = maxImpl;
    }

    
    public StorelessUnivariateStatistic getSumLogImpl() {
        return sumLogImpl;
    }

    
    public void setSumLogImpl(StorelessUnivariateStatistic sumLogImpl)
    throws MathIllegalStateException {
        checkEmpty();
        this.sumLogImpl = sumLogImpl;
        geoMean.setSumLogImpl(sumLogImpl);
    }

    
    public StorelessUnivariateStatistic getGeoMeanImpl() {
        return geoMeanImpl;
    }

    
    public void setGeoMeanImpl(StorelessUnivariateStatistic geoMeanImpl)
    throws MathIllegalStateException {
        checkEmpty();
        this.geoMeanImpl = geoMeanImpl;
    }

    
    public StorelessUnivariateStatistic getMeanImpl() {
        return meanImpl;
    }

    
    public void setMeanImpl(StorelessUnivariateStatistic meanImpl)
    throws MathIllegalStateException {
        checkEmpty();
        this.meanImpl = meanImpl;
    }

    
    public StorelessUnivariateStatistic getVarianceImpl() {
        return varianceImpl;
    }

    
    public void setVarianceImpl(StorelessUnivariateStatistic varianceImpl)
    throws MathIllegalStateException {
        checkEmpty();
        this.varianceImpl = varianceImpl;
    }

    
    private void checkEmpty() throws MathIllegalStateException {
        if (n > 0) {
            throw new MathIllegalStateException(
                LocalizedFormats.VALUES_ADDED_BEFORE_CONFIGURING_STATISTIC, n);
        }
    }

    
    public SummaryStatistics copy() {
        SummaryStatistics result = new SummaryStatistics();
        // No try-catch or advertised exception because arguments are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(SummaryStatistics source, SummaryStatistics dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.maxImpl = source.maxImpl.copy();
        dest.minImpl = source.minImpl.copy();
        dest.sumImpl = source.sumImpl.copy();
        dest.sumLogImpl = source.sumLogImpl.copy();
        dest.sumsqImpl = source.sumsqImpl.copy();
        dest.secondMoment = source.secondMoment.copy();
        dest.n = source.n;

        // Keep commons-math supplied statistics with embedded moments in synch
        if (source.getVarianceImpl() instanceof Variance) {
            dest.varianceImpl = new Variance(dest.secondMoment);
        } else {
            dest.varianceImpl = source.varianceImpl.copy();
        }
        if (source.meanImpl instanceof Mean) {
            dest.meanImpl = new Mean(dest.secondMoment);
        } else {
            dest.meanImpl = source.meanImpl.copy();
        }
        if (source.getGeoMeanImpl() instanceof GeometricMean) {
            dest.geoMeanImpl = new GeometricMean((SumOfLogs) dest.sumLogImpl);
        } else {
            dest.geoMeanImpl = source.geoMeanImpl.copy();
        }

        // Make sure that if stat == statImpl in source, same
        // holds in dest; otherwise copy stat
        if (source.geoMean == source.geoMeanImpl) {
            dest.geoMean = (GeometricMean) dest.geoMeanImpl;
        } else {
            GeometricMean.copy(source.geoMean, dest.geoMean);
        }
        if (source.max == source.maxImpl) {
            dest.max = (Max) dest.maxImpl;
        } else {
            Max.copy(source.max, dest.max);
        }
        if (source.mean == source.meanImpl) {
            dest.mean = (Mean) dest.meanImpl;
        } else {
            Mean.copy(source.mean, dest.mean);
        }
        if (source.min == source.minImpl) {
            dest.min = (Min) dest.minImpl;
        } else {
            Min.copy(source.min, dest.min);
        }
        if (source.sum == source.sumImpl) {
            dest.sum = (Sum) dest.sumImpl;
        } else {
            Sum.copy(source.sum, dest.sum);
        }
        if (source.variance == source.varianceImpl) {
            dest.variance = (Variance) dest.varianceImpl;
        } else {
            Variance.copy(source.variance, dest.variance);
        }
        if (source.sumLog == source.sumLogImpl) {
            dest.sumLog = (SumOfLogs) dest.sumLogImpl;
        } else {
            SumOfLogs.copy(source.sumLog, dest.sumLog);
        }
        if (source.sumsq == source.sumsqImpl) {
            dest.sumsq = (SumOfSquares) dest.sumsqImpl;
        } else {
            SumOfSquares.copy(source.sumsq, dest.sumsq);
        }
    }
}
