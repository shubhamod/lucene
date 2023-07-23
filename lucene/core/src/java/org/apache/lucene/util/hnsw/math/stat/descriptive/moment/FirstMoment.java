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

import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


class FirstMoment extends AbstractStorelessUnivariateStatistic
    implements Serializable {

    
    private static final long serialVersionUID = 6112755307178490473L;


    
    protected long n;

    
    protected double m1;

    
    protected double dev;

    
    protected double nDev;

    
    FirstMoment() {
        n = 0;
        m1 = Double.NaN;
        dev = Double.NaN;
        nDev = Double.NaN;
    }

    
     FirstMoment(FirstMoment original) throws NullArgumentException {
         super();
         copy(original, this);
     }

    
     @Override
    public void increment(final double d) {
        if (n == 0) {
            m1 = 0.0;
        }
        n++;
        double n0 = n;
        dev = d - m1;
        nDev = dev / n0;
        m1 += nDev;
    }

    
    @Override
    public void clear() {
        m1 = Double.NaN;
        n = 0;
        dev = Double.NaN;
        nDev = Double.NaN;
    }

    
    @Override
    public double getResult() {
        return m1;
    }

    
    public long getN() {
        return n;
    }

    
    @Override
    public FirstMoment copy() {
        FirstMoment result = new FirstMoment();
        // No try-catch or advertised exception because args are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(FirstMoment source, FirstMoment dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.n = source.n;
        dest.m1 = source.m1;
        dest.dev = source.dev;
        dest.nDev = source.nDev;
    }
}
