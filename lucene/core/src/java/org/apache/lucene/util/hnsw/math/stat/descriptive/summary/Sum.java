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
package org.apache.lucene.util.hnsw.math.stat.descriptive.summary;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.util.MathUtils;



public class Sum extends AbstractStorelessUnivariateStatistic implements Serializable {

    
    private static final long serialVersionUID = -8231831954703408316L;

    
    private long n;

    
    private double value;

    
    public Sum() {
        n = 0;
        value = 0;
    }

    
    public Sum(Sum original) throws NullArgumentException {
        copy(original, this);
    }

    
    @Override
    public void increment(final double d) {
        value += d;
        n++;
    }

    
    @Override
    public double getResult() {
        return value;
    }

    
    public long getN() {
        return n;
    }

    
    @Override
    public void clear() {
        value = 0;
        n = 0;
    }

    
    @Override
    public double evaluate(final double[] values, final int begin, final int length)
    throws MathIllegalArgumentException {
        double sum = Double.NaN;
        if (test(values, begin, length, true)) {
            sum = 0.0;
            for (int i = begin; i < begin + length; i++) {
                sum += values[i];
            }
        }
        return sum;
    }

    
    public double evaluate(final double[] values, final double[] weights,
        final int begin, final int length) throws MathIllegalArgumentException {
        double sum = Double.NaN;
        if (test(values, weights, begin, length, true)) {
            sum = 0.0;
            for (int i = begin; i < begin + length; i++) {
                sum += values[i] * weights[i];
            }
        }
        return sum;
    }

    
    public double evaluate(final double[] values, final double[] weights)
    throws MathIllegalArgumentException {
        return evaluate(values, weights, 0, values.length);
    }

    
    @Override
    public Sum copy() {
        Sum result = new Sum();
        // No try-catch or advertised exception because args are valid
        copy(this, result);
        return result;
    }

    
    public static void copy(Sum source, Sum dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.n = source.n;
        dest.value = source.value;
    }

}
