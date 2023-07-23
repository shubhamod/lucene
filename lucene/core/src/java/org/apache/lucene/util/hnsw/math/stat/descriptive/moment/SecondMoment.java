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


public class SecondMoment extends FirstMoment implements Serializable {

    
    private static final long serialVersionUID = 3942403127395076445L;

    
    protected double m2;

    
    public SecondMoment() {
        super();
        m2 = Double.NaN;
    }

    
    public SecondMoment(SecondMoment original)
    throws NullArgumentException {
        super(original);
        this.m2 = original.m2;
    }

    
    @Override
    public void increment(final double d) {
        if (n < 1) {
            m1 = m2 = 0.0;
        }
        super.increment(d);
        m2 += ((double) n - 1) * dev * nDev;
    }

    
    @Override
    public void clear() {
        super.clear();
        m2 = Double.NaN;
    }

    
    @Override
    public double getResult() {
        return m2;
    }

    
    @Override
    public SecondMoment copy() {
        SecondMoment result = new SecondMoment();
        // no try-catch or advertised NAE because args are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(SecondMoment source, SecondMoment dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        FirstMoment.copy(source, dest);
        dest.m2 = source.m2;
    }

}
