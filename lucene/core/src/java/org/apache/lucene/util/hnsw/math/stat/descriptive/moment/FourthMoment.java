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


class FourthMoment extends ThirdMoment implements Serializable{

    
    private static final long serialVersionUID = 4763990447117157611L;

    
    private double m4;

    
    FourthMoment() {
        super();
        m4 = Double.NaN;
    }

    
     FourthMoment(FourthMoment original) throws NullArgumentException {
         super();
         copy(original, this);
     }

    
     @Override
    public void increment(final double d) {
        if (n < 1) {
            m4 = 0.0;
            m3 = 0.0;
            m2 = 0.0;
            m1 = 0.0;
        }

        double prevM3 = m3;
        double prevM2 = m2;

        super.increment(d);

        double n0 = n;

        m4 = m4 - 4.0 * nDev * prevM3 + 6.0 * nDevSq * prevM2 +
            ((n0 * n0) - 3 * (n0 -1)) * (nDevSq * nDevSq * (n0 - 1) * n0);
    }

    
    @Override
    public double getResult() {
        return m4;
    }

    
    @Override
    public void clear() {
        super.clear();
        m4 = Double.NaN;
    }

    
    @Override
    public FourthMoment copy() {
        FourthMoment result = new FourthMoment();
        // No try-catch or advertised exception because args are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(FourthMoment source, FourthMoment dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        ThirdMoment.copy(source, dest);
        dest.m4 = source.m4;
    }
}
