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
package org.apache.lucene.util.hnsw.math.stat;

import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.stat.descriptive.DescriptiveStatistics;
import org.apache.lucene.util.hnsw.math.stat.descriptive.UnivariateStatistic;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.GeometricMean;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Mean;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Variance;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Max;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Min;
import org.apache.lucene.util.hnsw.math.stat.descriptive.rank.Percentile;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.Product;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.Sum;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.SumOfLogs;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.SumOfSquares;


public final class StatUtils {

    
    private static final UnivariateStatistic SUM = new Sum();

    
    private static final UnivariateStatistic SUM_OF_SQUARES = new SumOfSquares();

    
    private static final UnivariateStatistic PRODUCT = new Product();

    
    private static final UnivariateStatistic SUM_OF_LOGS = new SumOfLogs();

    
    private static final UnivariateStatistic MIN = new Min();

    
    private static final UnivariateStatistic MAX = new Max();

    
    private static final UnivariateStatistic MEAN = new Mean();

    
    private static final Variance VARIANCE = new Variance();

    
    private static final Percentile PERCENTILE = new Percentile();

    
    private static final GeometricMean GEOMETRIC_MEAN = new GeometricMean();

    
    private StatUtils() {
    }

    
    public static double sum(final double[] values)
    throws MathIllegalArgumentException {
        return SUM.evaluate(values);
    }

    
    public static double sum(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return SUM.evaluate(values, begin, length);
    }

    
    public static double sumSq(final double[] values) throws MathIllegalArgumentException {
        return SUM_OF_SQUARES.evaluate(values);
    }

    
    public static double sumSq(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return SUM_OF_SQUARES.evaluate(values, begin, length);
    }

    
    public static double product(final double[] values)
    throws MathIllegalArgumentException {
        return PRODUCT.evaluate(values);
    }

    
    public static double product(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return PRODUCT.evaluate(values, begin, length);
    }

    
    public static double sumLog(final double[] values)
    throws MathIllegalArgumentException {
        return SUM_OF_LOGS.evaluate(values);
    }

    
    public static double sumLog(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return SUM_OF_LOGS.evaluate(values, begin, length);
    }

    
    public static double mean(final double[] values)
    throws MathIllegalArgumentException {
        return MEAN.evaluate(values);
    }

    
    public static double mean(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return MEAN.evaluate(values, begin, length);
    }

    
    public static double geometricMean(final double[] values)
    throws MathIllegalArgumentException {
        return GEOMETRIC_MEAN.evaluate(values);
    }

    
    public static double geometricMean(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return GEOMETRIC_MEAN.evaluate(values, begin, length);
    }


    
    public static double variance(final double[] values) throws MathIllegalArgumentException {
        return VARIANCE.evaluate(values);
    }

    
    public static double variance(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return VARIANCE.evaluate(values, begin, length);
    }

    
    public static double variance(final double[] values, final double mean,
            final int begin, final int length) throws MathIllegalArgumentException {
        return VARIANCE.evaluate(values, mean, begin, length);
    }

    
    public static double variance(final double[] values, final double mean)
    throws MathIllegalArgumentException {
        return VARIANCE.evaluate(values, mean);
    }

    
    public static double populationVariance(final double[] values)
    throws MathIllegalArgumentException {
        return new Variance(false).evaluate(values);
    }

    
    public static double populationVariance(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return new Variance(false).evaluate(values, begin, length);
    }

    
    public static double populationVariance(final double[] values, final double mean,
            final int begin, final int length) throws MathIllegalArgumentException {
        return new Variance(false).evaluate(values, mean, begin, length);
    }

    
    public static double populationVariance(final double[] values, final double mean)
    throws MathIllegalArgumentException {
        return new Variance(false).evaluate(values, mean);
    }

    
    public static double max(final double[] values) throws MathIllegalArgumentException {
        return MAX.evaluate(values);
    }

    
    public static double max(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return MAX.evaluate(values, begin, length);
    }

     
    public static double min(final double[] values) throws MathIllegalArgumentException {
        return MIN.evaluate(values);
    }

     
    public static double min(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        return MIN.evaluate(values, begin, length);
    }

    
    public static double percentile(final double[] values, final double p)
    throws MathIllegalArgumentException {
            return PERCENTILE.evaluate(values,p);
    }

     
    public static double percentile(final double[] values, final int begin,
            final int length, final double p) throws MathIllegalArgumentException {
        return PERCENTILE.evaluate(values, begin, length, p);
    }

    
    public static double sumDifference(final double[] sample1, final double[] sample2)
    throws DimensionMismatchException, NoDataException {
        int n = sample1.length;
        if (n != sample2.length) {
            throw new DimensionMismatchException(n, sample2.length);
        }
        if (n <= 0) {
            throw new NoDataException(LocalizedFormats.INSUFFICIENT_DIMENSION);
        }
        double result = 0;
        for (int i = 0; i < n; i++) {
            result += sample1[i] - sample2[i];
        }
        return result;
    }

    
    public static double meanDifference(final double[] sample1, final double[] sample2)
    throws DimensionMismatchException, NoDataException{
        return sumDifference(sample1, sample2) / sample1.length;
    }

    
    public static double varianceDifference(final double[] sample1,
            final double[] sample2, double meanDifference) throws DimensionMismatchException,
            NumberIsTooSmallException {
        double sum1 = 0d;
        double sum2 = 0d;
        double diff = 0d;
        int n = sample1.length;
        if (n != sample2.length) {
            throw new DimensionMismatchException(n, sample2.length);
        }
        if (n < 2) {
            throw new NumberIsTooSmallException(n, 2, true);
        }
        for (int i = 0; i < n; i++) {
            diff = sample1[i] - sample2[i];
            sum1 += (diff - meanDifference) *(diff - meanDifference);
            sum2 += diff - meanDifference;
        }
        return (sum1 - (sum2 * sum2 / n)) / (n - 1);
    }

    
    public static double[] normalize(final double[] sample) {
        DescriptiveStatistics stats = new DescriptiveStatistics();

        // Add the data from the series to stats
        for (int i = 0; i < sample.length; i++) {
            stats.addValue(sample[i]);
        }

        // Compute mean and standard deviation
        double mean = stats.getMean();
        double standardDeviation = stats.getStandardDeviation();

        // initialize the standardizedSample, which has the same length as the sample
        double[] standardizedSample = new double[sample.length];

        for (int i = 0; i < sample.length; i++) {
            // z = (x- mean)/standardDeviation
            standardizedSample[i] = (sample[i] - mean) / standardDeviation;
        }
        return standardizedSample;
    }

    
    public static double[] mode(double[] sample) throws MathIllegalArgumentException {
        if (sample == null) {
            throw new NullArgumentException(LocalizedFormats.INPUT_ARRAY);
        }
        return getMode(sample, 0, sample.length);
    }

    
    public static double[] mode(double[] sample, final int begin, final int length) {
        if (sample == null) {
            throw new NullArgumentException(LocalizedFormats.INPUT_ARRAY);
        }

        if (begin < 0) {
            throw new NotPositiveException(LocalizedFormats.START_POSITION, Integer.valueOf(begin));
        }

        if (length < 0) {
            throw new NotPositiveException(LocalizedFormats.LENGTH, Integer.valueOf(length));
        }

        return getMode(sample, begin, length);
    }

    
    private static double[] getMode(double[] values, final int begin, final int length) {
        // Add the values to the frequency table
        Frequency freq = new Frequency();
        for (int i = begin; i < begin + length; i++) {
            final double value = values[i];
            if (!Double.isNaN(value)) {
                freq.addValue(Double.valueOf(value));
            }
        }
        List<Comparable<?>> list = freq.getMode();
        // Convert the list to an array of primitive double
        double[] modes = new double[list.size()];
        int i = 0;
        for(Comparable<?> c : list) {
            modes[i++] = ((Double) c).doubleValue();
        }
        return modes;
    }

}
