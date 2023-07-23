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
package org.apache.lucene.util.hnsw.math.stat.inference;

import java.util.Collection;

import org.apache.lucene.util.hnsw.math.distribution.RealDistribution;
import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.InsufficientDataException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.stat.descriptive.StatisticalSummary;


public class TestUtils  {

    
    private static final TTest T_TEST = new TTest();

    
    private static final ChiSquareTest CHI_SQUARE_TEST = new ChiSquareTest();

    
    private static final OneWayAnova ONE_WAY_ANANOVA = new OneWayAnova();

    
    private static final GTest G_TEST = new GTest();

    
    private static final KolmogorovSmirnovTest KS_TEST = new KolmogorovSmirnovTest();

    
    private TestUtils() {
        super();
    }

    // CHECKSTYLE: stop JavadocMethodCheck

    
    public static double homoscedasticT(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NumberIsTooSmallException {
        return T_TEST.homoscedasticT(sample1, sample2);
    }

    
    public static double homoscedasticT(final StatisticalSummary sampleStats1,
                                        final StatisticalSummary sampleStats2)
        throws NullArgumentException, NumberIsTooSmallException {
        return T_TEST.homoscedasticT(sampleStats1, sampleStats2);
    }

    
    public static boolean homoscedasticTTest(final double[] sample1, final double[] sample2,
                                             final double alpha)
        throws NullArgumentException, NumberIsTooSmallException,
        OutOfRangeException, MaxCountExceededException {
        return T_TEST.homoscedasticTTest(sample1, sample2, alpha);
    }

    
    public static double homoscedasticTTest(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NumberIsTooSmallException, MaxCountExceededException {
        return T_TEST.homoscedasticTTest(sample1, sample2);
    }

    
    public static double homoscedasticTTest(final StatisticalSummary sampleStats1,
                                            final StatisticalSummary sampleStats2)
        throws NullArgumentException, NumberIsTooSmallException, MaxCountExceededException {
        return T_TEST.homoscedasticTTest(sampleStats1, sampleStats2);
    }

    
    public static double pairedT(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NoDataException,
        DimensionMismatchException, NumberIsTooSmallException {
        return T_TEST.pairedT(sample1, sample2);
    }

    
    public static boolean pairedTTest(final double[] sample1, final double[] sample2,
                                      final double alpha)
        throws NullArgumentException, NoDataException, DimensionMismatchException,
        NumberIsTooSmallException, OutOfRangeException, MaxCountExceededException {
        return T_TEST.pairedTTest(sample1, sample2, alpha);
    }

    
    public static double pairedTTest(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NoDataException, DimensionMismatchException,
        NumberIsTooSmallException, MaxCountExceededException {
        return T_TEST.pairedTTest(sample1, sample2);
    }

    
    public static double t(final double mu, final double[] observed)
        throws NullArgumentException, NumberIsTooSmallException {
        return T_TEST.t(mu, observed);
    }

    
    public static double t(final double mu, final StatisticalSummary sampleStats)
        throws NullArgumentException, NumberIsTooSmallException {
        return T_TEST.t(mu, sampleStats);
    }

    
    public static double t(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NumberIsTooSmallException {
        return T_TEST.t(sample1, sample2);
    }

    
    public static double t(final StatisticalSummary sampleStats1,
                           final StatisticalSummary sampleStats2)
        throws NullArgumentException, NumberIsTooSmallException {
        return T_TEST.t(sampleStats1, sampleStats2);
    }

    
    public static boolean tTest(final double mu, final double[] sample, final double alpha)
        throws NullArgumentException, NumberIsTooSmallException,
        OutOfRangeException, MaxCountExceededException {
        return T_TEST.tTest(mu, sample, alpha);
    }

    
    public static double tTest(final double mu, final double[] sample)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {
        return T_TEST.tTest(mu, sample);
    }

    
    public static boolean tTest(final double mu, final StatisticalSummary sampleStats,
                                final double alpha)
        throws NullArgumentException, NumberIsTooSmallException,
        OutOfRangeException, MaxCountExceededException {
        return T_TEST.tTest(mu, sampleStats, alpha);
    }

    
    public static double tTest(final double mu, final StatisticalSummary sampleStats)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {
        return T_TEST.tTest(mu, sampleStats);
    }

    
    public static boolean tTest(final double[] sample1, final double[] sample2,
                                final double alpha)
        throws NullArgumentException, NumberIsTooSmallException,
        OutOfRangeException, MaxCountExceededException {
        return T_TEST.tTest(sample1, sample2, alpha);
    }

    
    public static double tTest(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {
        return T_TEST.tTest(sample1, sample2);
    }

    
    public static boolean tTest(final StatisticalSummary sampleStats1,
                                final StatisticalSummary sampleStats2,
                                final double alpha)
        throws NullArgumentException, NumberIsTooSmallException,
        OutOfRangeException, MaxCountExceededException {
        return T_TEST.tTest(sampleStats1, sampleStats2, alpha);
    }

    
    public static double tTest(final StatisticalSummary sampleStats1,
                               final StatisticalSummary sampleStats2)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {
        return T_TEST.tTest(sampleStats1, sampleStats2);
    }

    
    public static double chiSquare(final double[] expected, final long[] observed)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException {
        return CHI_SQUARE_TEST.chiSquare(expected, observed);
    }

    
    public static double chiSquare(final long[][] counts)
        throws NullArgumentException, NotPositiveException,
        DimensionMismatchException {
        return CHI_SQUARE_TEST.chiSquare(counts);
    }

    
    public static boolean chiSquareTest(final double[] expected, final long[] observed,
                                        final double alpha)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException, OutOfRangeException, MaxCountExceededException {
        return CHI_SQUARE_TEST.chiSquareTest(expected, observed, alpha);
    }

    
    public static double chiSquareTest(final double[] expected, final long[] observed)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException, MaxCountExceededException {
        return CHI_SQUARE_TEST.chiSquareTest(expected, observed);
    }

    
    public static boolean chiSquareTest(final long[][] counts, final double alpha)
        throws NullArgumentException, DimensionMismatchException,
        NotPositiveException, OutOfRangeException, MaxCountExceededException {
        return CHI_SQUARE_TEST.chiSquareTest(counts, alpha);
    }

    
    public static double chiSquareTest(final long[][] counts)
        throws NullArgumentException, DimensionMismatchException,
        NotPositiveException, MaxCountExceededException {
        return CHI_SQUARE_TEST.chiSquareTest(counts);
    }

    
    public static double chiSquareDataSetsComparison(final long[] observed1,
                                                     final long[] observed2)
        throws DimensionMismatchException, NotPositiveException, ZeroException {
        return CHI_SQUARE_TEST.chiSquareDataSetsComparison(observed1, observed2);
    }

    
    public static double chiSquareTestDataSetsComparison(final long[] observed1,
                                                         final long[] observed2)
        throws DimensionMismatchException, NotPositiveException, ZeroException,
        MaxCountExceededException {
        return CHI_SQUARE_TEST.chiSquareTestDataSetsComparison(observed1, observed2);
    }

    
    public static boolean chiSquareTestDataSetsComparison(final long[] observed1,
                                                          final long[] observed2,
                                                          final double alpha)
        throws DimensionMismatchException, NotPositiveException,
        ZeroException, OutOfRangeException, MaxCountExceededException {
        return CHI_SQUARE_TEST.chiSquareTestDataSetsComparison(observed1, observed2, alpha);
    }

    
    public static double oneWayAnovaFValue(final Collection<double[]> categoryData)
        throws NullArgumentException, DimensionMismatchException {
        return ONE_WAY_ANANOVA.anovaFValue(categoryData);
    }

    
    public static double oneWayAnovaPValue(final Collection<double[]> categoryData)
        throws NullArgumentException, DimensionMismatchException,
        ConvergenceException, MaxCountExceededException {
        return ONE_WAY_ANANOVA.anovaPValue(categoryData);
    }

    
    public static boolean oneWayAnovaTest(final Collection<double[]> categoryData,
                                          final double alpha)
        throws NullArgumentException, DimensionMismatchException,
        OutOfRangeException, ConvergenceException, MaxCountExceededException {
        return ONE_WAY_ANANOVA.anovaTest(categoryData, alpha);
    }

     
    public static double g(final double[] expected, final long[] observed)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException {
        return G_TEST.g(expected, observed);
    }

    
    public static double gTest(final double[] expected, final long[] observed)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException, MaxCountExceededException {
        return G_TEST.gTest(expected, observed);
    }

    
    public static double gTestIntrinsic(final double[] expected, final long[] observed)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException, MaxCountExceededException {
        return G_TEST.gTestIntrinsic(expected, observed);
    }

     
    public static boolean gTest(final double[] expected, final long[] observed,
                                final double alpha)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException, OutOfRangeException, MaxCountExceededException {
        return G_TEST.gTest(expected, observed, alpha);
    }

    
    public static double gDataSetsComparison(final long[] observed1,
                                                  final long[] observed2)
        throws DimensionMismatchException, NotPositiveException, ZeroException {
        return G_TEST.gDataSetsComparison(observed1, observed2);
    }

    
    public static double rootLogLikelihoodRatio(final long k11, final long k12, final long k21, final long k22)
        throws DimensionMismatchException, NotPositiveException, ZeroException {
        return G_TEST.rootLogLikelihoodRatio(k11, k12, k21, k22);
    }


    
    public static double gTestDataSetsComparison(final long[] observed1,
                                                        final long[] observed2)
        throws DimensionMismatchException, NotPositiveException, ZeroException,
        MaxCountExceededException {
        return G_TEST.gTestDataSetsComparison(observed1, observed2);
    }

    
    public static boolean gTestDataSetsComparison(final long[] observed1,
                                                  final long[] observed2,
                                                  final double alpha)
        throws DimensionMismatchException, NotPositiveException,
        ZeroException, OutOfRangeException, MaxCountExceededException {
        return G_TEST.gTestDataSetsComparison(observed1, observed2, alpha);
    }

    
    public static double kolmogorovSmirnovStatistic(RealDistribution dist, double[] data)
            throws InsufficientDataException, NullArgumentException {
        return KS_TEST.kolmogorovSmirnovStatistic(dist, data);
    }

    
    public static double kolmogorovSmirnovTest(RealDistribution dist, double[] data)
            throws InsufficientDataException, NullArgumentException {
        return KS_TEST.kolmogorovSmirnovTest(dist, data);
    }

    
    public static double kolmogorovSmirnovTest(RealDistribution dist, double[] data, boolean strict)
            throws InsufficientDataException, NullArgumentException {
        return KS_TEST.kolmogorovSmirnovTest(dist, data, strict);
    }

    
    public static boolean kolmogorovSmirnovTest(RealDistribution dist, double[] data, double alpha)
            throws InsufficientDataException, NullArgumentException {
        return KS_TEST.kolmogorovSmirnovTest(dist, data, alpha);
    }

    
    public static double kolmogorovSmirnovStatistic(double[] x, double[] y)
            throws InsufficientDataException, NullArgumentException {
        return KS_TEST.kolmogorovSmirnovStatistic(x, y);
    }

    
    public static double kolmogorovSmirnovTest(double[] x, double[] y)
            throws InsufficientDataException, NullArgumentException {
        return KS_TEST.kolmogorovSmirnovTest(x, y);
    }

    
    public static double kolmogorovSmirnovTest(double[] x, double[] y, boolean strict)
            throws InsufficientDataException, NullArgumentException  {
        return KS_TEST.kolmogorovSmirnovTest(x, y, strict);
    }

    
    public static double exactP(double d, int m, int n, boolean strict) {
        return KS_TEST.exactP(d, n, m, strict);
    }

    
    public static double approximateP(double d, int n, int m) {
        return KS_TEST.approximateP(d, n, m);
    }

    
    public static double monteCarloP(double d, int n, int m, boolean strict, int iterations) {
        return KS_TEST.monteCarloP(d, n, m, strict, iterations);
    }


    // CHECKSTYLE: resume JavadocMethodCheck

}
