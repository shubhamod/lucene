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
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.stat.descriptive.WeightedEvaluation;
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class Variance extends AbstractStorelessUnivariateStatistic implements Serializable, WeightedEvaluation {

    
    private static final long serialVersionUID = -9111962718267217978L;

    
    protected SecondMoment moment = null;

    
    protected boolean incMoment = true;

    
    private boolean isBiasCorrected = true;

    
    public Variance() {
        moment = new SecondMoment();
    }

    
    public Variance(final SecondMoment m2) {
        incMoment = false;
        this.moment = m2;
    }

    
    public Variance(boolean isBiasCorrected) {
        moment = new SecondMoment();
        this.isBiasCorrected = isBiasCorrected;
    }

    
    public Variance(boolean isBiasCorrected, SecondMoment m2) {
        incMoment = false;
        this.moment = m2;
        this.isBiasCorrected = isBiasCorrected;
    }

    
    public Variance(Variance original) throws NullArgumentException {
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
            if (moment.n == 0) {
                return Double.NaN;
            } else if (moment.n == 1) {
                return 0d;
            } else {
                if (isBiasCorrected) {
                    return moment.m2 / (moment.n - 1d);
                } else {
                    return moment.m2 / (moment.n);
                }
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
    public double evaluate(final double[] values) throws MathIllegalArgumentException {
        if (values == null) {
            throw new NullArgumentException(LocalizedFormats.INPUT_ARRAY);
        }
        return evaluate(values, 0, values.length);
    }

    
    @Override
    public double evaluate(final double[] values, final int begin, final int length)
    throws MathIllegalArgumentException {

        double var = Double.NaN;

        if (test(values, begin, length)) {
            clear();
            if (length == 1) {
                var = 0.0;
            } else if (length > 1) {
                Mean mean = new Mean();
                double m = mean.evaluate(values, begin, length);
                var = evaluate(values, m, begin, length);
            }
        }
        return var;
    }

    
    public double evaluate(final double[] values, final double[] weights,
                           final int begin, final int length) throws MathIllegalArgumentException {

        double var = Double.NaN;

        if (test(values, weights,begin, length)) {
            clear();
            if (length == 1) {
                var = 0.0;
            } else if (length > 1) {
                Mean mean = new Mean();
                double m = mean.evaluate(values, weights, begin, length);
                var = evaluate(values, weights, m, begin, length);
            }
        }
        return var;
    }

    
    public double evaluate(final double[] values, final double[] weights)
    throws MathIllegalArgumentException {
        return evaluate(values, weights, 0, values.length);
    }

    
    public double evaluate(final double[] values, final double mean,
            final int begin, final int length) throws MathIllegalArgumentException {

        double var = Double.NaN;

        if (test(values, begin, length)) {
            if (length == 1) {
                var = 0.0;
            } else if (length > 1) {
                double accum = 0.0;
                double dev = 0.0;
                double accum2 = 0.0;
                for (int i = begin; i < begin + length; i++) {
                    dev = values[i] - mean;
                    accum += dev * dev;
                    accum2 += dev;
                }
                double len = length;
                if (isBiasCorrected) {
                    var = (accum - (accum2 * accum2 / len)) / (len - 1.0);
                } else {
                    var = (accum - (accum2 * accum2 / len)) / len;
                }
            }
        }
        return var;
    }

    
    public double evaluate(final double[] values, final double mean) throws MathIllegalArgumentException {
        return evaluate(values, mean, 0, values.length);
    }

    
    public double evaluate(final double[] values, final double[] weights,
    final double mean, final int begin, final int length)
    throws MathIllegalArgumentException {

        double var = Double.NaN;

        if (test(values, weights, begin, length)) {
            if (length == 1) {
                var = 0.0;
            } else if (length > 1) {
                double accum = 0.0;
                double dev = 0.0;
                double accum2 = 0.0;
                for (int i = begin; i < begin + length; i++) {
                    dev = values[i] - mean;
                    accum += weights[i] * (dev * dev);
                    accum2 += weights[i] * dev;
                }

                double sumWts = 0;
                for (int i = begin; i < begin + length; i++) {
                    sumWts += weights[i];
                }

                if (isBiasCorrected) {
                    var = (accum - (accum2 * accum2 / sumWts)) / (sumWts - 1.0);
                } else {
                    var = (accum - (accum2 * accum2 / sumWts)) / sumWts;
                }
            }
        }
        return var;
    }

    
    public double evaluate(final double[] values, final double[] weights, final double mean)
    throws MathIllegalArgumentException {
        return evaluate(values, weights, mean, 0, values.length);
    }

    
    public boolean isBiasCorrected() {
        return isBiasCorrected;
    }

    
    public void setBiasCorrected(boolean biasCorrected) {
        this.isBiasCorrected = biasCorrected;
    }

    
    @Override
    public Variance copy() {
        Variance result = new Variance();
        // No try-catch or advertised exception because parameters are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(Variance source, Variance dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.moment = source.moment.copy();
        dest.isBiasCorrected = source.isBiasCorrected;
        dest.incMoment = source.incMoment;
    }
}
