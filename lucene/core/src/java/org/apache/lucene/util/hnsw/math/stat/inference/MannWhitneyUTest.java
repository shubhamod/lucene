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
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.stat.ranking.NaNStrategy;
import org.apache.lucene.util.hnsw.math.stat.ranking.NaturalRanking;
import org.apache.lucene.util.hnsw.math.stat.ranking.TiesStrategy;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class MannWhitneyUTest {

    
    private NaturalRanking naturalRanking;

    
    public MannWhitneyUTest() {
        naturalRanking = new NaturalRanking(NaNStrategy.FIXED,
                TiesStrategy.AVERAGE);
    }

    
    public MannWhitneyUTest(final NaNStrategy nanStrategy,
                            final TiesStrategy tiesStrategy) {
        naturalRanking = new NaturalRanking(nanStrategy, tiesStrategy);
    }

    
    private void ensureDataConformance(final double[] x, final double[] y)
        throws NullArgumentException, NoDataException {

        if (x == null ||
            y == null) {
            throw new NullArgumentException();
        }
        if (x.length == 0 ||
            y.length == 0) {
            throw new NoDataException();
        }
    }

    
    private double[] concatenateSamples(final double[] x, final double[] y) {
        final double[] z = new double[x.length + y.length];

        System.arraycopy(x, 0, z, 0, x.length);
        System.arraycopy(y, 0, z, x.length, y.length);

        return z;
    }

    
    public double mannWhitneyU(final double[] x, final double[] y)
        throws NullArgumentException, NoDataException {

        ensureDataConformance(x, y);

        final double[] z = concatenateSamples(x, y);
        final double[] ranks = naturalRanking.rank(z);

        double sumRankX = 0;

        /*
         * The ranks for x is in the first x.length entries in ranks because x
         * is in the first x.length entries in z
         */
        for (int i = 0; i < x.length; ++i) {
            sumRankX += ranks[i];
        }

        /*
         * U1 = R1 - (n1 * (n1 + 1)) / 2 where R1 is sum of ranks for sample 1,
         * e.g. x, n1 is the number of observations in sample 1.
         */
        final double U1 = sumRankX - ((long) x.length * (x.length + 1)) / 2;

        /*
         * It can be shown that U1 + U2 = n1 * n2
         */
        final double U2 = (long) x.length * y.length - U1;

        return FastMath.max(U1, U2);
    }

    
    private double calculateAsymptoticPValue(final double Umin,
                                             final int n1,
                                             final int n2)
        throws ConvergenceException, MaxCountExceededException {

        /* long multiplication to avoid overflow (double not used due to efficiency
         * and to avoid precision loss)
         */
        final long n1n2prod = (long) n1 * n2;

        // http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Normal_approximation
        final double EU = n1n2prod / 2.0;
        final double VarU = n1n2prod * (n1 + n2 + 1) / 12.0;

        final double z = (Umin - EU) / FastMath.sqrt(VarU);

        // No try-catch or advertised exception because args are valid
        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final NormalDistribution standardNormal = new NormalDistribution(null, 0, 1);

        return 2 * standardNormal.cumulativeProbability(z);
    }

    
    public double mannWhitneyUTest(final double[] x, final double[] y)
        throws NullArgumentException, NoDataException,
        ConvergenceException, MaxCountExceededException {

        ensureDataConformance(x, y);

        final double Umax = mannWhitneyU(x, y);

        /*
         * It can be shown that U1 + U2 = n1 * n2
         */
        final double Umin = (long) x.length * y.length - Umax;

        return calculateAsymptoticPValue(Umin, x.length, y.length);
    }

}
