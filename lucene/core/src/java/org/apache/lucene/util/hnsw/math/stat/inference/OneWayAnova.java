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

import java.util.ArrayList;
import java.util.Collection;

import org.apache.lucene.util.hnsw.math.distribution.FDistribution;
import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.stat.descriptive.SummaryStatistics;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class OneWayAnova {

    
    public OneWayAnova() {
    }

    
    public double anovaFValue(final Collection<double[]> categoryData)
        throws NullArgumentException, DimensionMismatchException {

        AnovaStats a = anovaStats(categoryData);
        return a.F;

    }

    
    public double anovaPValue(final Collection<double[]> categoryData)
        throws NullArgumentException, DimensionMismatchException,
        ConvergenceException, MaxCountExceededException {

        final AnovaStats a = anovaStats(categoryData);
        // No try-catch or advertised exception because args are valid
        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final FDistribution fdist = new FDistribution(null, a.dfbg, a.dfwg);
        return 1.0 - fdist.cumulativeProbability(a.F);

    }

    
    public double anovaPValue(final Collection<SummaryStatistics> categoryData,
                              final boolean allowOneElementData)
        throws NullArgumentException, DimensionMismatchException,
               ConvergenceException, MaxCountExceededException {

        final AnovaStats a = anovaStats(categoryData, allowOneElementData);
        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final FDistribution fdist = new FDistribution(null, a.dfbg, a.dfwg);
        return 1.0 - fdist.cumulativeProbability(a.F);

    }

    
    private AnovaStats anovaStats(final Collection<double[]> categoryData)
        throws NullArgumentException, DimensionMismatchException {

        MathUtils.checkNotNull(categoryData);

        final Collection<SummaryStatistics> categoryDataSummaryStatistics =
                new ArrayList<SummaryStatistics>(categoryData.size());

        // convert arrays to SummaryStatistics
        for (final double[] data : categoryData) {
            final SummaryStatistics dataSummaryStatistics = new SummaryStatistics();
            categoryDataSummaryStatistics.add(dataSummaryStatistics);
            for (final double val : data) {
                dataSummaryStatistics.addValue(val);
            }
        }

        return anovaStats(categoryDataSummaryStatistics, false);

    }

    
    public boolean anovaTest(final Collection<double[]> categoryData,
                             final double alpha)
        throws NullArgumentException, DimensionMismatchException,
        OutOfRangeException, ConvergenceException, MaxCountExceededException {

        if ((alpha <= 0) || (alpha > 0.5)) {
            throw new OutOfRangeException(
                    LocalizedFormats.OUT_OF_BOUND_SIGNIFICANCE_LEVEL,
                    alpha, 0, 0.5);
        }
        return anovaPValue(categoryData) < alpha;

    }

    
    private AnovaStats anovaStats(final Collection<SummaryStatistics> categoryData,
                                  final boolean allowOneElementData)
        throws NullArgumentException, DimensionMismatchException {

        MathUtils.checkNotNull(categoryData);

        if (!allowOneElementData) {
            // check if we have enough categories
            if (categoryData.size() < 2) {
                throw new DimensionMismatchException(LocalizedFormats.TWO_OR_MORE_CATEGORIES_REQUIRED,
                                                     categoryData.size(), 2);
            }

            // check if each category has enough data
            for (final SummaryStatistics array : categoryData) {
                if (array.getN() <= 1) {
                    throw new DimensionMismatchException(LocalizedFormats.TWO_OR_MORE_VALUES_IN_CATEGORY_REQUIRED,
                                                         (int) array.getN(), 2);
                }
            }
        }

        int dfwg = 0;
        double sswg = 0;
        double totsum = 0;
        double totsumsq = 0;
        int totnum = 0;

        for (final SummaryStatistics data : categoryData) {

            final double sum = data.getSum();
            final double sumsq = data.getSumsq();
            final int num = (int) data.getN();
            totnum += num;
            totsum += sum;
            totsumsq += sumsq;

            dfwg += num - 1;
            final double ss = sumsq - ((sum * sum) / num);
            sswg += ss;
        }

        final double sst = totsumsq - ((totsum * totsum) / totnum);
        final double ssbg = sst - sswg;
        final int dfbg = categoryData.size() - 1;
        final double msbg = ssbg / dfbg;
        final double mswg = sswg / dfwg;
        final double F = msbg / mswg;

        return new AnovaStats(dfbg, dfwg, F);

    }

    
    private static class AnovaStats {

        
        private final int dfbg;

        
        private final int dfwg;

        
        private final double F;

        
        private AnovaStats(int dfbg, int dfwg, double F) {
            this.dfbg = dfbg;
            this.dfwg = dfwg;
            this.F = F;
        }
    }

}
