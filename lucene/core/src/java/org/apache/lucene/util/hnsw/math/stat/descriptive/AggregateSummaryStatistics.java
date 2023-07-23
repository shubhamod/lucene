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
import java.util.Collection;
import java.util.Iterator;

import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;


public class AggregateSummaryStatistics implements StatisticalSummary,
        Serializable {


    
    private static final long serialVersionUID = -8207112444016386906L;

    
    private final SummaryStatistics statisticsPrototype;

    
    private final SummaryStatistics statistics;

    
    public AggregateSummaryStatistics() {
        // No try-catch or throws NAE because arg is guaranteed non-null
        this(new SummaryStatistics());
    }

    
    public AggregateSummaryStatistics(SummaryStatistics prototypeStatistics) throws NullArgumentException {
        this(prototypeStatistics,
             prototypeStatistics == null ? null : new SummaryStatistics(prototypeStatistics));
    }

    
    public AggregateSummaryStatistics(SummaryStatistics prototypeStatistics,
                                      SummaryStatistics initialStatistics) {
        this.statisticsPrototype =
            (prototypeStatistics == null) ? new SummaryStatistics() : prototypeStatistics;
        this.statistics =
            (initialStatistics == null) ? new SummaryStatistics() : initialStatistics;
    }

    
    public double getMax() {
        synchronized (statistics) {
            return statistics.getMax();
        }
    }

    
    public double getMean() {
        synchronized (statistics) {
            return statistics.getMean();
        }
    }

    
    public double getMin() {
        synchronized (statistics) {
            return statistics.getMin();
        }
    }

    
    public long getN() {
        synchronized (statistics) {
            return statistics.getN();
        }
    }

    
    public double getStandardDeviation() {
        synchronized (statistics) {
            return statistics.getStandardDeviation();
        }
    }

    
    public double getSum() {
        synchronized (statistics) {
            return statistics.getSum();
        }
    }

    
    public double getVariance() {
        synchronized (statistics) {
            return statistics.getVariance();
        }
    }

    
    public double getSumOfLogs() {
        synchronized (statistics) {
            return statistics.getSumOfLogs();
        }
    }

    
    public double getGeometricMean() {
        synchronized (statistics) {
            return statistics.getGeometricMean();
        }
    }

    
    public double getSumsq() {
        synchronized (statistics) {
            return statistics.getSumsq();
        }
    }

    
    public double getSecondMoment() {
        synchronized (statistics) {
            return statistics.getSecondMoment();
        }
    }

    
    public StatisticalSummary getSummary() {
        synchronized (statistics) {
            return new StatisticalSummaryValues(getMean(), getVariance(), getN(),
                    getMax(), getMin(), getSum());
        }
    }

    
    public SummaryStatistics createContributingStatistics() {
        SummaryStatistics contributingStatistics
                = new AggregatingSummaryStatistics(statistics);

        // No try - catch or advertising NAE because neither argument will ever be null
        SummaryStatistics.copy(statisticsPrototype, contributingStatistics);

        return contributingStatistics;
    }

    
    public static StatisticalSummaryValues aggregate(Collection<? extends StatisticalSummary> statistics) {
        if (statistics == null) {
            return null;
        }
        Iterator<? extends StatisticalSummary> iterator = statistics.iterator();
        if (!iterator.hasNext()) {
            return null;
        }
        StatisticalSummary current = iterator.next();
        long n = current.getN();
        double min = current.getMin();
        double sum = current.getSum();
        double max = current.getMax();
        double var = current.getVariance();
        double m2 = var * (n - 1d);
        double mean = current.getMean();
        while (iterator.hasNext()) {
            current = iterator.next();
            if (current.getMin() < min || Double.isNaN(min)) {
                min = current.getMin();
            }
            if (current.getMax() > max || Double.isNaN(max)) {
                max = current.getMax();
            }
            sum += current.getSum();
            final double oldN = n;
            final double curN = current.getN();
            n += curN;
            final double meanDiff = current.getMean() - mean;
            mean = sum / n;
            final double curM2 = current.getVariance() * (curN - 1d);
            m2 = m2 + curM2 + meanDiff * meanDiff * oldN * curN / n;
        }
        final double variance;
        if (n == 0) {
            variance = Double.NaN;
        } else if (n == 1) {
            variance = 0d;
        } else {
            variance = m2 / (n - 1);
        }
        return new StatisticalSummaryValues(mean, variance, n, max, min, sum);
    }

    
    private static class AggregatingSummaryStatistics extends SummaryStatistics {

        
        private static final long serialVersionUID = 1L;

        
        private final SummaryStatistics aggregateStatistics;

        
        AggregatingSummaryStatistics(SummaryStatistics aggregateStatistics) {
            this.aggregateStatistics = aggregateStatistics;
        }

        
        @Override
        public void addValue(double value) {
            super.addValue(value);
            synchronized (aggregateStatistics) {
                aggregateStatistics.addValue(value);
            }
        }

        
        @Override
        public boolean equals(Object object) {
            if (object == this) {
                return true;
            }
            if (object instanceof AggregatingSummaryStatistics == false) {
                return false;
            }
            AggregatingSummaryStatistics stat = (AggregatingSummaryStatistics)object;
            return super.equals(stat) &&
                   aggregateStatistics.equals(stat.aggregateStatistics);
        }

        
        @Override
        public int hashCode() {
            return 123 + super.hashCode() + aggregateStatistics.hashCode();
        }
    }
}
