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
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.GeometricMean;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Kurtosis;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Mean;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Skewness;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Variance;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Max;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Min;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Percentile;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.Sum;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.SumOfSquares;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.ResizableDoubleArray;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class DescriptiveStatistics implements StatisticalSummary, Serializable {

    
    public static final int INFINITE_WINDOW = -1;

    
    private static final long serialVersionUID = 4133067267405273064L;

    
    private static final String SET_QUANTILE_METHOD_NAME = "setQuantile";

    
    protected int windowSize = INFINITE_WINDOW;

    
    private ResizableDoubleArray eDA = new ResizableDoubleArray();

    
    private UnivariateStatistic meanImpl = new Mean();

    
    private UnivariateStatistic geometricMeanImpl = new GeometricMean();

    
    private UnivariateStatistic kurtosisImpl = new Kurtosis();

    
    private UnivariateStatistic maxImpl = new Max();

    
    private UnivariateStatistic minImpl = new Min();

    
    private UnivariateStatistic percentileImpl = new Percentile();

    
    private UnivariateStatistic skewnessImpl = new Skewness();

    
    private UnivariateStatistic varianceImpl = new Variance();

    
    private UnivariateStatistic sumsqImpl = new SumOfSquares();

    
    private UnivariateStatistic sumImpl = new Sum();

    
    public DescriptiveStatistics() {
    }

    
    public DescriptiveStatistics(int window) throws MathIllegalArgumentException {
        setWindowSize(window);
    }

    
    public DescriptiveStatistics(double[] initialDoubleArray) {
        if (initialDoubleArray != null) {
            eDA = new ResizableDoubleArray(initialDoubleArray);
        }
    }

    
    public DescriptiveStatistics(DescriptiveStatistics original) throws NullArgumentException {
        copy(original, this);
    }

    
    public void addValue(double v) {
        if (windowSize != INFINITE_WINDOW) {
            if (getN() == windowSize) {
                eDA.addElementRolling(v);
            } else if (getN() < windowSize) {
                eDA.addElement(v);
            }
        } else {
            eDA.addElement(v);
        }
    }

    
    public void removeMostRecentValue() throws MathIllegalStateException {
        try {
            eDA.discardMostRecentElements(1);
        } catch (MathIllegalArgumentException ex) {
            throw new MathIllegalStateException(LocalizedFormats.NO_DATA);
        }
    }

    
    public double replaceMostRecentValue(double v) throws MathIllegalStateException {
        return eDA.substituteMostRecentElement(v);
    }

    
    public double getMean() {
        return apply(meanImpl);
    }

    
    public double getGeometricMean() {
        return apply(geometricMeanImpl);
    }

    
    public double getVariance() {
        return apply(varianceImpl);
    }

    
    public double getPopulationVariance() {
        return apply(new Variance(false));
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
        final long n = getN();
        return n > 0 ? FastMath.sqrt(getSumsq() / n) : Double.NaN;
    }

    
    public double getSkewness() {
        return apply(skewnessImpl);
    }

    
    public double getKurtosis() {
        return apply(kurtosisImpl);
    }

    
    public double getMax() {
        return apply(maxImpl);
    }

    
    public double getMin() {
        return apply(minImpl);
    }

    
    public long getN() {
        return eDA.getNumElements();
    }

    
    public double getSum() {
        return apply(sumImpl);
    }

    
    public double getSumsq() {
        return apply(sumsqImpl);
    }

    
    public void clear() {
        eDA.clear();
    }


    
    public int getWindowSize() {
        return windowSize;
    }

    
    public void setWindowSize(int windowSize) throws MathIllegalArgumentException {
        if (windowSize < 1 && windowSize != INFINITE_WINDOW) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.NOT_POSITIVE_WINDOW_SIZE, windowSize);
        }

        this.windowSize = windowSize;

        // We need to check to see if we need to discard elements
        // from the front of the array.  If the windowSize is less than
        // the current number of elements.
        if (windowSize != INFINITE_WINDOW && windowSize < eDA.getNumElements()) {
            eDA.discardFrontElements(eDA.getNumElements() - windowSize);
        }
    }

    
    public double[] getValues() {
        return eDA.getElements();
    }

    
    public double[] getSortedValues() {
        double[] sort = getValues();
        Arrays.sort(sort);
        return sort;
    }

    
    public double getElement(int index) {
        return eDA.getElement(index);
    }

    
    public double getPercentile(double p) throws MathIllegalStateException, MathIllegalArgumentException {
        if (percentileImpl instanceof Percentile) {
            ((Percentile) percentileImpl).setQuantile(p);
        } else {
            try {
                percentileImpl.getClass().getMethod(SET_QUANTILE_METHOD_NAME,
                        new Class[] {Double.TYPE}).invoke(percentileImpl,
                                new Object[] {Double.valueOf(p)});
            } catch (NoSuchMethodException e1) { // Setter guard should prevent
                throw new MathIllegalStateException(
                      LocalizedFormats.PERCENTILE_IMPLEMENTATION_UNSUPPORTED_METHOD,
                      percentileImpl.getClass().getName(), SET_QUANTILE_METHOD_NAME);
            } catch (IllegalAccessException e2) {
                throw new MathIllegalStateException(
                      LocalizedFormats.PERCENTILE_IMPLEMENTATION_CANNOT_ACCESS_METHOD,
                      SET_QUANTILE_METHOD_NAME, percentileImpl.getClass().getName());
            } catch (InvocationTargetException e3) {
                throw new IllegalStateException(e3.getCause());
            }
        }
        return apply(percentileImpl);
    }

    
    @Override
    public String toString() {
        StringBuilder outBuffer = new StringBuilder();
        String endl = "\n";
        outBuffer.append("DescriptiveStatistics:").append(endl);
        outBuffer.append("n: ").append(getN()).append(endl);
        outBuffer.append("min: ").append(getMin()).append(endl);
        outBuffer.append("max: ").append(getMax()).append(endl);
        outBuffer.append("mean: ").append(getMean()).append(endl);
        outBuffer.append("std dev: ").append(getStandardDeviation())
            .append(endl);
        try {
            // No catch for MIAE because actual parameter is valid below
            outBuffer.append("median: ").append(getPercentile(50)).append(endl);
        } catch (MathIllegalStateException ex) {
            outBuffer.append("median: unavailable").append(endl);
        }
        outBuffer.append("skewness: ").append(getSkewness()).append(endl);
        outBuffer.append("kurtosis: ").append(getKurtosis()).append(endl);
        return outBuffer.toString();
    }

    
    public double apply(UnivariateStatistic stat) {
        // No try-catch or advertised exception here because arguments are guaranteed valid
        return eDA.compute(stat);
    }

    // Implementation getters and setter

    
    public synchronized UnivariateStatistic getMeanImpl() {
        return meanImpl;
    }

    
    public synchronized void setMeanImpl(UnivariateStatistic meanImpl) {
        this.meanImpl = meanImpl;
    }

    
    public synchronized UnivariateStatistic getGeometricMeanImpl() {
        return geometricMeanImpl;
    }

    
    public synchronized void setGeometricMeanImpl(
            UnivariateStatistic geometricMeanImpl) {
        this.geometricMeanImpl = geometricMeanImpl;
    }

    
    public synchronized UnivariateStatistic getKurtosisImpl() {
        return kurtosisImpl;
    }

    
    public synchronized void setKurtosisImpl(UnivariateStatistic kurtosisImpl) {
        this.kurtosisImpl = kurtosisImpl;
    }

    
    public synchronized UnivariateStatistic getMaxImpl() {
        return maxImpl;
    }

    
    public synchronized void setMaxImpl(UnivariateStatistic maxImpl) {
        this.maxImpl = maxImpl;
    }

    
    public synchronized UnivariateStatistic getMinImpl() {
        return minImpl;
    }

    
    public synchronized void setMinImpl(UnivariateStatistic minImpl) {
        this.minImpl = minImpl;
    }

    
    public synchronized UnivariateStatistic getPercentileImpl() {
        return percentileImpl;
    }

    
    public synchronized void setPercentileImpl(UnivariateStatistic percentileImpl)
    throws MathIllegalArgumentException {
        try {
            percentileImpl.getClass().getMethod(SET_QUANTILE_METHOD_NAME,
                    new Class[] {Double.TYPE}).invoke(percentileImpl,
                            new Object[] {Double.valueOf(50.0d)});
        } catch (NoSuchMethodException e1) {
            throw new MathIllegalArgumentException(
                  LocalizedFormats.PERCENTILE_IMPLEMENTATION_UNSUPPORTED_METHOD,
                  percentileImpl.getClass().getName(), SET_QUANTILE_METHOD_NAME);
        } catch (IllegalAccessException e2) {
            throw new MathIllegalArgumentException(
                  LocalizedFormats.PERCENTILE_IMPLEMENTATION_CANNOT_ACCESS_METHOD,
                  SET_QUANTILE_METHOD_NAME, percentileImpl.getClass().getName());
        } catch (InvocationTargetException e3) {
            throw new IllegalArgumentException(e3.getCause());
        }
        this.percentileImpl = percentileImpl;
    }

    
    public synchronized UnivariateStatistic getSkewnessImpl() {
        return skewnessImpl;
    }

    
    public synchronized void setSkewnessImpl(
            UnivariateStatistic skewnessImpl) {
        this.skewnessImpl = skewnessImpl;
    }

    
    public synchronized UnivariateStatistic getVarianceImpl() {
        return varianceImpl;
    }

    
    public synchronized void setVarianceImpl(
            UnivariateStatistic varianceImpl) {
        this.varianceImpl = varianceImpl;
    }

    
    public synchronized UnivariateStatistic getSumsqImpl() {
        return sumsqImpl;
    }

    
    public synchronized void setSumsqImpl(UnivariateStatistic sumsqImpl) {
        this.sumsqImpl = sumsqImpl;
    }

    
    public synchronized UnivariateStatistic getSumImpl() {
        return sumImpl;
    }

    
    public synchronized void setSumImpl(UnivariateStatistic sumImpl) {
        this.sumImpl = sumImpl;
    }

    
    public DescriptiveStatistics copy() {
        DescriptiveStatistics result = new DescriptiveStatistics();
        // No try-catch or advertised exception because parms are guaranteed valid
        copy(this, result);
        return result;
    }

    
    public static void copy(DescriptiveStatistics source, DescriptiveStatistics dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        // Copy data and window size
        dest.eDA = source.eDA.copy();
        dest.windowSize = source.windowSize;

        // Copy implementations
        dest.maxImpl = source.maxImpl.copy();
        dest.meanImpl = source.meanImpl.copy();
        dest.minImpl = source.minImpl.copy();
        dest.sumImpl = source.sumImpl.copy();
        dest.varianceImpl = source.varianceImpl.copy();
        dest.sumsqImpl = source.sumsqImpl.copy();
        dest.geometricMeanImpl = source.geometricMeanImpl.copy();
        dest.kurtosisImpl = source.kurtosisImpl;
        dest.skewnessImpl = source.skewnessImpl;
        dest.percentileImpl = source.percentileImpl;
    }
}
