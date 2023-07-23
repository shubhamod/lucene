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
package org.apache.lucene.util.hnsw.math.stat.descriptive.moment;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.stat.descriptive.WeightedEvaluation;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.Sum;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class Mean extends AbstractStorelessUnivariateStatistic
    implements Serializable, WeightedEvaluation {

    
    private static final long serialVersionUID = -1296043746617791564L;

    
    protected FirstMoment moment;

    
    protected boolean incMoment;

    
    public Mean() {
        incMoment = true;
        moment = new FirstMoment();
    }

    
    public Mean(final FirstMoment m1) {
        this.moment = m1;
        incMoment = false;
    }

    
    public Mean(Mean original) throws NullArgumentException {
        copy(original, this);
    }

    
    @Override
    public void increment(final double d) {
        if (incMoment) {
            moment.increment(d);
        }
    }

    
    @Override
    public void clear() {
        if (incMoment) {
            moment.clear();
        }
    }

    
    @Override
    public double getResult() {
        return moment.m1;
    }

    
    public long getN() {
        return moment.getN();
    }

    
    @Override
    public double evaluate(final double[] values,final int begin, final int length)
    throws MathIllegalArgumentException {
        if (test(values, begin, length)) {
            Sum sum = new Sum();
            double sampleSize = length;

            // Compute initial estimate using definitional formula
            double xbar = sum.evaluate(values, begin, length) / sampleSize;

            // Compute correction factor in second pass
            double correction = 0;
            for (int i = begin; i < begin + length; i++) {
                correction += values[i] - xbar;
            }
            return xbar + (correction/sampleSize);
        }
        return Double.NaN;
    }

    
    public double evaluate(final double[] values, final double[] weights,
                           final int begin, final int length) throws MathIllegalArgumentException {
        if (test(values, weights, begin, length)) {
            Sum sum = new Sum();

            // Compute initial estimate using definitional formula
            double sumw = sum.evaluate(weights,begin,length);
            double xbarw = sum.evaluate(values, weights, begin, length) / sumw;

            // Compute correction factor in second pass
            double correction = 0;
            for (int i = begin; i < begin + length; i++) {
                correction += weights[i] * (values[i] - xbarw);
            }
            return xbarw + (correction/sumw);
        }
        return Double.NaN;
    }

    
    public double evaluate(final double[] values, final double[] weights)
    throws MathIllegalArgumentException {
        return evaluate(values, weights, 0, values.length);
    }

    
    @Override
    public Mean copy() {
        Mean result = new Mean();
        // No try-catch or advertised exception because args are guaranteed non-null
        copy(this, result);
        return result;
    }


    
    public static void copy(Mean source, Mean dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.incMoment = source.incMoment;
        dest.moment = source.moment.copy();
    }
}
