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

import org.apache.lucene.util.hnsw.math.distribution.ChiSquaredDistribution;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class ChiSquareTest {

    
    public ChiSquareTest() {
        super();
    }

    
    public double chiSquare(final double[] expected, final long[] observed)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException {

        if (expected.length < 2) {
            throw new DimensionMismatchException(expected.length, 2);
        }
        if (expected.length != observed.length) {
            throw new DimensionMismatchException(expected.length, observed.length);
        }
        MathArrays.checkPositive(expected);
        MathArrays.checkNonNegative(observed);

        double sumExpected = 0d;
        double sumObserved = 0d;
        for (int i = 0; i < observed.length; i++) {
            sumExpected += expected[i];
            sumObserved += observed[i];
        }
        double ratio = 1.0d;
        boolean rescale = false;
        if (FastMath.abs(sumExpected - sumObserved) > 10E-6) {
            ratio = sumObserved / sumExpected;
            rescale = true;
        }
        double sumSq = 0.0d;
        for (int i = 0; i < observed.length; i++) {
            if (rescale) {
                final double dev = observed[i] - ratio * expected[i];
                sumSq += dev * dev / (ratio * expected[i]);
            } else {
                final double dev = observed[i] - expected[i];
                sumSq += dev * dev / expected[i];
            }
        }
        return sumSq;

    }

    
    public double chiSquareTest(final double[] expected, final long[] observed)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException, MaxCountExceededException {

        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final ChiSquaredDistribution distribution =
            new ChiSquaredDistribution(null, expected.length - 1.0);
        return 1.0 - distribution.cumulativeProbability(chiSquare(expected, observed));
    }

    
    public boolean chiSquareTest(final double[] expected, final long[] observed,
                                 final double alpha)
        throws NotPositiveException, NotStrictlyPositiveException,
        DimensionMismatchException, OutOfRangeException, MaxCountExceededException {

        if ((alpha <= 0) || (alpha > 0.5)) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_BOUND_SIGNIFICANCE_LEVEL,
                                          alpha, 0, 0.5);
        }
        return chiSquareTest(expected, observed) < alpha;

    }

    
    public double chiSquare(final long[][] counts)
        throws NullArgumentException, NotPositiveException,
        DimensionMismatchException {

        checkArray(counts);
        int nRows = counts.length;
        int nCols = counts[0].length;

        // compute row, column and total sums
        double[] rowSum = new double[nRows];
        double[] colSum = new double[nCols];
        double total = 0.0d;
        for (int row = 0; row < nRows; row++) {
            for (int col = 0; col < nCols; col++) {
                rowSum[row] += counts[row][col];
                colSum[col] += counts[row][col];
                total += counts[row][col];
            }
        }

        // compute expected counts and chi-square
        double sumSq = 0.0d;
        double expected = 0.0d;
        for (int row = 0; row < nRows; row++) {
            for (int col = 0; col < nCols; col++) {
                expected = (rowSum[row] * colSum[col]) / total;
                sumSq += ((counts[row][col] - expected) *
                        (counts[row][col] - expected)) / expected;
            }
        }
        return sumSq;

    }

    
    public double chiSquareTest(final long[][] counts)
        throws NullArgumentException, DimensionMismatchException,
        NotPositiveException, MaxCountExceededException {

        checkArray(counts);
        double df = ((double) counts.length -1) * ((double) counts[0].length - 1);
        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final ChiSquaredDistribution distribution = new ChiSquaredDistribution(df);
        return 1 - distribution.cumulativeProbability(chiSquare(counts));

    }

    
    public boolean chiSquareTest(final long[][] counts, final double alpha)
        throws NullArgumentException, DimensionMismatchException,
        NotPositiveException, OutOfRangeException, MaxCountExceededException {

        if ((alpha <= 0) || (alpha > 0.5)) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_BOUND_SIGNIFICANCE_LEVEL,
                                          alpha, 0, 0.5);
        }
        return chiSquareTest(counts) < alpha;

    }

    
    public double chiSquareDataSetsComparison(long[] observed1, long[] observed2)
        throws DimensionMismatchException, NotPositiveException, ZeroException {

        // Make sure lengths are same
        if (observed1.length < 2) {
            throw new DimensionMismatchException(observed1.length, 2);
        }
        if (observed1.length != observed2.length) {
            throw new DimensionMismatchException(observed1.length, observed2.length);
        }

        // Ensure non-negative counts
        MathArrays.checkNonNegative(observed1);
        MathArrays.checkNonNegative(observed2);

        // Compute and compare count sums
        long countSum1 = 0;
        long countSum2 = 0;
        boolean unequalCounts = false;
        double weight = 0.0;
        for (int i = 0; i < observed1.length; i++) {
            countSum1 += observed1[i];
            countSum2 += observed2[i];
        }
        // Ensure neither sample is uniformly 0
        if (countSum1 == 0 || countSum2 == 0) {
            throw new ZeroException();
        }
        // Compare and compute weight only if different
        unequalCounts = countSum1 != countSum2;
        if (unequalCounts) {
            weight = FastMath.sqrt((double) countSum1 / (double) countSum2);
        }
        // Compute ChiSquare statistic
        double sumSq = 0.0d;
        double dev = 0.0d;
        double obs1 = 0.0d;
        double obs2 = 0.0d;
        for (int i = 0; i < observed1.length; i++) {
            if (observed1[i] == 0 && observed2[i] == 0) {
                throw new ZeroException(LocalizedFormats.OBSERVED_COUNTS_BOTTH_ZERO_FOR_ENTRY, i);
            } else {
                obs1 = observed1[i];
                obs2 = observed2[i];
                if (unequalCounts) { // apply weights
                    dev = obs1/weight - obs2 * weight;
                } else {
                    dev = obs1 - obs2;
                }
                sumSq += (dev * dev) / (obs1 + obs2);
            }
        }
        return sumSq;
    }

    
    public double chiSquareTestDataSetsComparison(long[] observed1, long[] observed2)
        throws DimensionMismatchException, NotPositiveException, ZeroException,
        MaxCountExceededException {

        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final ChiSquaredDistribution distribution =
                new ChiSquaredDistribution(null, (double) observed1.length - 1);
        return 1 - distribution.cumulativeProbability(
                chiSquareDataSetsComparison(observed1, observed2));

    }

    
    public boolean chiSquareTestDataSetsComparison(final long[] observed1,
                                                   final long[] observed2,
                                                   final double alpha)
        throws DimensionMismatchException, NotPositiveException,
        ZeroException, OutOfRangeException, MaxCountExceededException {

        if (alpha <= 0 ||
            alpha > 0.5) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_BOUND_SIGNIFICANCE_LEVEL,
                                          alpha, 0, 0.5);
        }
        return chiSquareTestDataSetsComparison(observed1, observed2) < alpha;

    }

    
    private void checkArray(final long[][] in)
        throws NullArgumentException, DimensionMismatchException,
        NotPositiveException {

        if (in.length < 2) {
            throw new DimensionMismatchException(in.length, 2);
        }

        if (in[0].length < 2) {
            throw new DimensionMismatchException(in[0].length, 2);
        }

        MathArrays.checkRectangular(in);
        MathArrays.checkNonNegative(in);

    }

}
