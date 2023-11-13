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



public class Kurtosis extends AbstractStorelessUnivariateStatistic  implements Serializable {

    
    private static final long serialVersionUID = 2784465764798260919L;

    
    protected FourthMoment moment;

    
    protected boolean incMoment;

    
    public Kurtosis() {
        incMoment = true;
        moment = new FourthMoment();
    }

    
    public Kurtosis(final FourthMoment m4) {
        incMoment = false;
        this.moment = m4;
    }

    
    public Kurtosis(Kurtosis original) throws NullArgumentException {
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
        double kurtosis = Double.NaN;
        if (moment.getN() > 3) {
            double variance = moment.m2 / (moment.n - 1);
                if (moment.n <= 3 || variance < 10E-20) {
                    kurtosis = 0.0;
                } else {
                    double n = moment.n;
                    kurtosis =
                        (n * (n + 1) * moment.getResult() -
                                3 * moment.m2 * moment.m2 * (n - 1)) /
                                ((n - 1) * (n -2) * (n -3) * variance * variance);
                }
        }
        return kurtosis;
    }

    
    @Override
    public void clear() {
        if (incMoment) {
            moment.clear();
        }
    }

    
    public long getN() {
        return moment.getN();
    }

    /* UnvariateStatistic Approach  */

    
    @Override
    public double evaluate(final double[] values,final int begin, final int length)
    throws MathIllegalArgumentException {
        // Initialize the kurtosis
        double kurt = Double.NaN;

        if (test(values, begin, length) && length > 3) {

            // Compute the mean and standard deviation
            Variance variance = new Variance();
            variance.incrementAll(values, begin, length);
            double mean = variance.moment.m1;
            double stdDev = FastMath.sqrt(variance.getResult());

            // Sum the ^4 of the distance from the mean divided by the
            // standard deviation
            double accum3 = 0.0;
            for (int i = begin; i < begin + length; i++) {
                accum3 += FastMath.pow(values[i] - mean, 4.0);
            }
            accum3 /= FastMath.pow(stdDev, 4.0d);

            // Get N
            double n0 = length;

            double coefficientOne =
                (n0 * (n0 + 1)) / ((n0 - 1) * (n0 - 2) * (n0 - 3));
            double termTwo =
                (3 * FastMath.pow(n0 - 1, 2.0)) / ((n0 - 2) * (n0 - 3));

            // Calculate kurtosis
            kurt = (coefficientOne * accum3) - termTwo;
        }
        return kurt;
    }

    
    @Override
    public Kurtosis copy() {
        Kurtosis result = new Kurtosis();
        // No try-catch because args are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(Kurtosis source, Kurtosis dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.moment = source.moment.copy();
        dest.incMoment = source.incMoment;
    }

}
