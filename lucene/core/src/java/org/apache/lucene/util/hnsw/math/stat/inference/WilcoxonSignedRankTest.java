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

import org.apache.lucene.util.hnsw.math.distribution.NormalDistribution;
import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.stat.ranking.NaNStrategy;
import org.apache.lucene.util.hnsw.math.stat.ranking.NaturalRanking;
import org.apache.lucene.util.hnsw.math.stat.ranking.TiesStrategy;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class WilcoxonSignedRankTest {

    
    private NaturalRanking naturalRanking;

    
    public WilcoxonSignedRankTest() {
        naturalRanking = new NaturalRanking(NaNStrategy.FIXED,
                TiesStrategy.AVERAGE);
    }

    
    public WilcoxonSignedRankTest(final NaNStrategy nanStrategy,
                                  final TiesStrategy tiesStrategy) {
        naturalRanking = new NaturalRanking(nanStrategy, tiesStrategy);
    }

    
    private void ensureDataConformance(final double[] x, final double[] y)
        throws NullArgumentException, NoDataException, DimensionMismatchException {

        if (x == null ||
            y == null) {
                throw new NullArgumentException();
        }
        if (x.length == 0 ||
            y.length == 0) {
            throw new NoDataException();
        }
        if (y.length != x.length) {
            throw new DimensionMismatchException(y.length, x.length);
        }
    }

    
    private double[] calculateDifferences(final double[] x, final double[] y) {

        final double[] z = new double[x.length];

        for (int i = 0; i < x.length; ++i) {
            z[i] = y[i] - x[i];
        }

        return z;
    }

    
    private double[] calculateAbsoluteDifferences(final double[] z)
        throws NullArgumentException, NoDataException {

        if (z == null) {
            throw new NullArgumentException();
        }

        if (z.length == 0) {
            throw new NoDataException();
        }

        final double[] zAbs = new double[z.length];

        for (int i = 0; i < z.length; ++i) {
            zAbs[i] = FastMath.abs(z[i]);
        }

        return zAbs;
    }

    
    public double wilcoxonSignedRank(final double[] x, final double[] y)
        throws NullArgumentException, NoDataException, DimensionMismatchException {

        ensureDataConformance(x, y);

        // throws IllegalArgumentException if x and y are not correctly
        // specified
        final double[] z = calculateDifferences(x, y);
        final double[] zAbs = calculateAbsoluteDifferences(z);

        final double[] ranks = naturalRanking.rank(zAbs);

        double Wplus = 0;

        for (int i = 0; i < z.length; ++i) {
            if (z[i] > 0) {
                Wplus += ranks[i];
            }
        }

        final int N = x.length;
        final double Wminus = (((double) (N * (N + 1))) / 2.0) - Wplus;

        return FastMath.max(Wplus, Wminus);
    }

    
    private double calculateExactPValue(final double Wmax, final int N) {

        // Total number of outcomes (equal to 2^N but a lot faster)
        final int m = 1 << N;

        int largerRankSums = 0;

        for (int i = 0; i < m; ++i) {
            int rankSum = 0;

            // Generate all possible rank sums
            for (int j = 0; j < N; ++j) {

                // (i >> j) & 1 extract i's j-th bit from the right
                if (((i >> j) & 1) == 1) {
                    rankSum += j + 1;
                }
            }

            if (rankSum >= Wmax) {
                ++largerRankSums;
            }
        }

        /*
         * largerRankSums / m gives the one-sided p-value, so it's multiplied
         * with 2 to get the two-sided p-value
         */
        return 2 * ((double) largerRankSums) / ((double) m);
    }

    
    private double calculateAsymptoticPValue(final double Wmin, final int N) {

        final double ES = (double) (N * (N + 1)) / 4.0;

        /* Same as (but saves computations):
         * final double VarW = ((double) (N * (N + 1) * (2*N + 1))) / 24;
         */
        final double VarS = ES * ((double) (2 * N + 1) / 6.0);

        // - 0.5 is a continuity correction
        final double z = (Wmin - ES - 0.5) / FastMath.sqrt(VarS);

        // No try-catch or advertised exception because args are valid
        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final NormalDistribution standardNormal = new NormalDistribution(null, 0, 1);

        return 2*standardNormal.cumulativeProbability(z);
    }

    
    public double wilcoxonSignedRankTest(final double[] x, final double[] y,
                                         final boolean exactPValue)
        throws NullArgumentException, NoDataException, DimensionMismatchException,
        NumberIsTooLargeException, ConvergenceException, MaxCountExceededException {

        ensureDataConformance(x, y);

        final int N = x.length;
        final double Wmax = wilcoxonSignedRank(x, y);

        if (exactPValue && N > 30) {
            throw new NumberIsTooLargeException(N, 30, true);
        }

        if (exactPValue) {
            return calculateExactPValue(Wmax, N);
        } else {
            final double Wmin = ( (double)(N*(N+1)) / 2.0 ) - Wmax;
            return calculateAsymptoticPValue(Wmin, N);
        }
    }
}
