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
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class SumOfLogs extends AbstractStorelessUnivariateStatistic implements Serializable {

    
    private static final long serialVersionUID = -370076995648386763L;

    
    private int n;

    
    private double value;

    
    public SumOfLogs() {
       value = 0d;
       n = 0;
    }

    
    public SumOfLogs(SumOfLogs original) throws NullArgumentException {
        copy(original, this);
    }

    
    @Override
    public void increment(final double d) {
        value += FastMath.log(d);
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
        value = 0d;
        n = 0;
    }

    
    @Override
    public double evaluate(final double[] values, final int begin, final int length)
    throws MathIllegalArgumentException {
        double sumLog = Double.NaN;
        if (test(values, begin, length, true)) {
            sumLog = 0.0;
            for (int i = begin; i < begin + length; i++) {
                sumLog += FastMath.log(values[i]);
            }
        }
        return sumLog;
    }

    
    @Override
    public SumOfLogs copy() {
        SumOfLogs result = new SumOfLogs();
        // No try-catch or advertised exception here because args are valid
        copy(this, result);
        return result;
    }

    
    public static void copy(SumOfLogs source, SumOfLogs dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.n = source.n;
        dest.value = source.value;
    }
}
