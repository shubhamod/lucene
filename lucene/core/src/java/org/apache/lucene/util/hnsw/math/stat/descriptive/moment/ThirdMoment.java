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
import org.apache.lucene.util.hnsw.math.util.MathUtils;



class ThirdMoment extends SecondMoment implements Serializable {

    
    private static final long serialVersionUID = -7818711964045118679L;

    
    protected double m3;

     
    protected double nDevSq;

    
    ThirdMoment() {
        super();
        m3 = Double.NaN;
        nDevSq = Double.NaN;
    }

    
    ThirdMoment(ThirdMoment original) throws NullArgumentException {
        copy(original, this);
    }

    
    @Override
    public void increment(final double d) {
        if (n < 1) {
            m3 = m2 = m1 = 0.0;
        }

        double prevM2 = m2;
        super.increment(d);
        nDevSq = nDev * nDev;
        double n0 = n;
        m3 = m3 - 3.0 * nDev * prevM2 + (n0 - 1) * (n0 - 2) * nDevSq * dev;
    }

    
    @Override
    public double getResult() {
        return m3;
    }

    
    @Override
    public void clear() {
        super.clear();
        m3 = Double.NaN;
        nDevSq = Double.NaN;
    }

    
    @Override
    public ThirdMoment copy() {
        ThirdMoment result = new ThirdMoment();
        // No try-catch or advertised exception because args are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(ThirdMoment source, ThirdMoment dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        SecondMoment.copy(source, dest);
        dest.m3 = source.m3;
        dest.nDevSq = source.nDevSq;
    }

}
