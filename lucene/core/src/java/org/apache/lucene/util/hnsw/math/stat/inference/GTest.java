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
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class GTest {

    
    public double g(final double[] expected, final long[] observed)
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
        double ratio = 1d;
        boolean rescale = false;
        if (FastMath.abs(sumExpected - sumObserved) > 10E-6) {
            ratio = sumObserved / sumExpected;
            rescale = true;
        }
        double sum = 0d;
        for (int i = 0; i < observed.length; i++) {
            final double dev = rescale ?
                    FastMath.log((double) observed[i] / (ratio * expected[i])) :
                        FastMath.log((double) observed[i] / expected[i]);
            sum += ((double) observed[i]) * dev;
        }
        return 2d * sum;
    }

    
    public double gTest(final double[] expected, final long[] observed)
            throws NotPositiveException, NotStrictlyPositiveException,
            DimensionMismatchException, MaxCountExceededException {

        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final ChiSquaredDistribution distribution =
                new ChiSquaredDistribution(null, expected.length - 1.0);
        return 1.0 - distribution.cumulativeProbability(g(expected, observed));
    }

    
    public double gTestIntrinsic(final double[] expected, final long[] observed)
            throws NotPositiveException, NotStrictlyPositiveException,
            DimensionMismatchException, MaxCountExceededException {

        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final ChiSquaredDistribution distribution =
                new ChiSquaredDistribution(null, expected.length - 2.0);
        return 1.0 - distribution.cumulativeProbability(g(expected, observed));
    }

    
    public boolean gTest(final double[] expected, final long[] observed,
            final double alpha)
            throws NotPositiveException, NotStrictlyPositiveException,
            DimensionMismatchException, OutOfRangeException, MaxCountExceededException {

        if ((alpha <= 0) || (alpha > 0.5)) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_BOUND_SIGNIFICANCE_LEVEL,
                    alpha, 0, 0.5);
        }
        return gTest(expected, observed) < alpha;
    }

    
    private double entropy(final long[][] k) {
        double h = 0d;
        double sum_k = 0d;
        for (int i = 0; i < k.length; i++) {
            for (int j = 0; j < k[i].length; j++) {
                sum_k += (double) k[i][j];
            }
        }
        for (int i = 0; i < k.length; i++) {
            for (int j = 0; j < k[i].length; j++) {
                if (k[i][j] != 0) {
                    final double p_ij = (double) k[i][j] / sum_k;
                    h += p_ij * FastMath.log(p_ij);
                }
            }
        }
        return -h;
    }

    
    private double entropy(final long[] k) {
        double h = 0d;
        double sum_k = 0d;
        for (int i = 0; i < k.length; i++) {
            sum_k += (double) k[i];
        }
        for (int i = 0; i < k.length; i++) {
            if (k[i] != 0) {
                final double p_i = (double) k[i] / sum_k;
                h += p_i * FastMath.log(p_i);
            }
        }
        return -h;
    }

    
    public double gDataSetsComparison(final long[] observed1, final long[] observed2)
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

        // Compute and compare count sums
        final long[] collSums = new long[observed1.length];
        final long[][] k = new long[2][observed1.length];

        for (int i = 0; i < observed1.length; i++) {
            if (observed1[i] == 0 && observed2[i] == 0) {
                throw new ZeroException(LocalizedFormats.OBSERVED_COUNTS_BOTTH_ZERO_FOR_ENTRY, i);
            } else {
                countSum1 += observed1[i];
                countSum2 += observed2[i];
                collSums[i] = observed1[i] + observed2[i];
                k[0][i] = observed1[i];
                k[1][i] = observed2[i];
            }
        }
        // Ensure neither sample is uniformly 0
        if (countSum1 == 0 || countSum2 == 0) {
            throw new ZeroException();
        }
        final long[] rowSums = {countSum1, countSum2};
        final double sum = (double) countSum1 + (double) countSum2;
        return 2 * sum * (entropy(rowSums) + entropy(collSums) - entropy(k));
    }

    
    public double rootLogLikelihoodRatio(final long k11, long k12,
            final long k21, final long k22) {
        final double llr = gDataSetsComparison(
                new long[]{k11, k12}, new long[]{k21, k22});
        double sqrt = FastMath.sqrt(llr);
        if ((double) k11 / (k11 + k12) < (double) k21 / (k21 + k22)) {
            sqrt = -sqrt;
        }
        return sqrt;
    }

    
    public double gTestDataSetsComparison(final long[] observed1,
            final long[] observed2)
            throws DimensionMismatchException, NotPositiveException, ZeroException,
            MaxCountExceededException {

        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final ChiSquaredDistribution distribution =
                new ChiSquaredDistribution(null, (double) observed1.length - 1);
        return 1 - distribution.cumulativeProbability(
                gDataSetsComparison(observed1, observed2));
    }

    
    public boolean gTestDataSetsComparison(
            final long[] observed1,
            final long[] observed2,
            final double alpha)
            throws DimensionMismatchException, NotPositiveException,
            ZeroException, OutOfRangeException, MaxCountExceededException {

        if (alpha <= 0 || alpha > 0.5) {
            throw new OutOfRangeException(
                    LocalizedFormats.OUT_OF_BOUND_SIGNIFICANCE_LEVEL, alpha, 0, 0.5);
        }
        return gTestDataSetsComparison(observed1, observed2) < alpha;
    }
}
