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
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class Skewness extends AbstractStorelessUnivariateStatistic implements Serializable {

    
    private static final long serialVersionUID = 7101857578996691352L;

    
    protected ThirdMoment moment = null;

     
    protected boolean incMoment;

    
    public Skewness() {
        incMoment = true;
        moment = new ThirdMoment();
    }

    
    public Skewness(final ThirdMoment m3) {
        incMoment = false;
        this.moment = m3;
    }

    
    public Skewness(Skewness original) throws NullArgumentException {
        copy(original, this);
    }

    
    @Override
    public void increment(final double d) {
        if (incMoment) {
            moment.increment(d);
        }
    }

    
    @Override
    public double getResult() {

        if (moment.n < 3) {
            return Double.NaN;
        }
        double variance = moment.m2 / (moment.n - 1);
        if (variance < 10E-20) {
            return 0.0d;
        } else {
            double n0 = moment.getN();
            return  (n0 * moment.m3) /
            ((n0 - 1) * (n0 -2) * FastMath.sqrt(variance) * variance);
        }
    }

    
    public long getN() {
        return moment.getN();
    }

    
    @Override
    public void clear() {
        if (incMoment) {
            moment.clear();
        }
    }

    
    @Override
    public double evaluate(final double[] values,final int begin,
            final int length) throws MathIllegalArgumentException {

        // Initialize the skewness
        double skew = Double.NaN;

        if (test(values, begin, length) && length > 2 ){
            Mean mean = new Mean();
            // Get the mean and the standard deviation
            double m = mean.evaluate(values, begin, length);

            // Calc the std, this is implemented here instead
            // of using the standardDeviation method eliminate
            // a duplicate pass to get the mean
            double accum = 0.0;
            double accum2 = 0.0;
            for (int i = begin; i < begin + length; i++) {
                final double d = values[i] - m;
                accum  += d * d;
                accum2 += d;
            }
            final double variance = (accum - (accum2 * accum2 / length)) / (length - 1);

            double accum3 = 0.0;
            for (int i = begin; i < begin + length; i++) {
                final double d = values[i] - m;
                accum3 += d * d * d;
            }
            accum3 /= variance * FastMath.sqrt(variance);

            // Get N
            double n0 = length;

            // Calculate skewness
            skew = (n0 / ((n0 - 1) * (n0 - 2))) * accum3;
        }
        return skew;
    }

    
    @Override
    public Skewness copy() {
        Skewness result = new Skewness();
        // No try-catch or advertised exception because args are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(Skewness source, Skewness dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.moment = new ThirdMoment(source.moment.copy());
        dest.incMoment = source.incMoment;
    }
}
