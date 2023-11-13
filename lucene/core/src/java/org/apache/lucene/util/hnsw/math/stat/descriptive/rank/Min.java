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
package org.apache.lucene.util.hnsw.math.stat.descriptive.rank;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class Min extends AbstractStorelessUnivariateStatistic implements Serializable {

    
    private static final long serialVersionUID = -2941995784909003131L;

    
    private long n;

    
    private double value;

    
    public Min() {
        n = 0;
        value = Double.NaN;
    }

    
    public Min(Min original) throws NullArgumentException {
        copy(original, this);
    }

    
    @Override
    public void increment(final double d) {
        if (d < value || Double.isNaN(value)) {
            value = d;
        }
        n++;
    }

    
    @Override
    public void clear() {
        value = Double.NaN;
        n = 0;
    }

    
    @Override
    public double getResult() {
        return value;
    }

    
    public long getN() {
        return n;
    }

    
    @Override
    public double evaluate(final double[] values,final int begin, final int length)
    throws MathIllegalArgumentException {
        double min = Double.NaN;
        if (test(values, begin, length)) {
            min = values[begin];
            for (int i = begin; i < begin + length; i++) {
                if (!Double.isNaN(values[i])) {
                    min = (min < values[i]) ? min : values[i];
                }
            }
        }
        return min;
    }

    
    @Override
    public Min copy() {
        Min result = new Min();
        // No try-catch or advertised exception - args are non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(Min source, Min dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.n = source.n;
        dest.value = source.value;
    }
}
