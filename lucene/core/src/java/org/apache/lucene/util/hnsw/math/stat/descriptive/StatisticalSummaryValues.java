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

import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class StatisticalSummaryValues implements Serializable,
    StatisticalSummary {

    
    private static final long serialVersionUID = -5108854841843722536L;

    
    private final double mean;

    
    private final double variance;

    
    private final long n;

    
    private final double max;

    
    private final double min;

    
    private final double sum;

    
    public StatisticalSummaryValues(double mean, double variance, long n,
        double max, double min, double sum) {
        super();
        this.mean = mean;
        this.variance = variance;
        this.n = n;
        this.max = max;
        this.min = min;
        this.sum = sum;
    }

    
    public double getMax() {
        return max;
    }

    
    public double getMean() {
        return mean;
    }

    
    public double getMin() {
        return min;
    }

    
    public long getN() {
        return n;
    }

    
    public double getSum() {
        return sum;
    }

    
    public double getStandardDeviation() {
        return FastMath.sqrt(variance);
    }

    
    public double getVariance() {
        return variance;
    }

    
    @Override
    public boolean equals(Object object) {
        if (object == this ) {
            return true;
        }
        if (object instanceof StatisticalSummaryValues == false) {
            return false;
        }
        StatisticalSummaryValues stat = (StatisticalSummaryValues) object;
        return Precision.equalsIncludingNaN(stat.getMax(),      getMax())  &&
               Precision.equalsIncludingNaN(stat.getMean(),     getMean()) &&
               Precision.equalsIncludingNaN(stat.getMin(),      getMin())  &&
               Precision.equalsIncludingNaN(stat.getN(),        getN())    &&
               Precision.equalsIncludingNaN(stat.getSum(),      getSum())  &&
               Precision.equalsIncludingNaN(stat.getVariance(), getVariance());
    }

    
    @Override
    public int hashCode() {
        int result = 31 + MathUtils.hash(getMax());
        result = result * 31 + MathUtils.hash(getMean());
        result = result * 31 + MathUtils.hash(getMin());
        result = result * 31 + MathUtils.hash(getN());
        result = result * 31 + MathUtils.hash(getSum());
        result = result * 31 + MathUtils.hash(getVariance());
        return result;
    }

    
    @Override
    public String toString() {
        StringBuffer outBuffer = new StringBuffer();
        String endl = "\n";
        outBuffer.append("StatisticalSummaryValues:").append(endl);
        outBuffer.append("n: ").append(getN()).append(endl);
        outBuffer.append("min: ").append(getMin()).append(endl);
        outBuffer.append("max: ").append(getMax()).append(endl);
        outBuffer.append("mean: ").append(getMean()).append(endl);
        outBuffer.append("std dev: ").append(getStandardDeviation())
            .append(endl);
        outBuffer.append("variance: ").append(getVariance()).append(endl);
        outBuffer.append("sum: ").append(getSum()).append(endl);
        return outBuffer.toString();
    }

}
