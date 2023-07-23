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
package org.apache.lucene.util.hnsw.math.stat.descriptive;

import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.Precision;


public abstract class AbstractStorelessUnivariateStatistic
    extends AbstractUnivariateStatistic
    implements StorelessUnivariateStatistic {

    
    @Override
    public double evaluate(final double[] values) throws MathIllegalArgumentException {
        if (values == null) {
            throw new NullArgumentException(LocalizedFormats.INPUT_ARRAY);
        }
        return evaluate(values, 0, values.length);
    }

    
    @Override
    public double evaluate(final double[] values, final int begin,
            final int length) throws MathIllegalArgumentException {
        if (test(values, begin, length)) {
            clear();
            incrementAll(values, begin, length);
        }
        return getResult();
    }

    
    @Override
    public abstract StorelessUnivariateStatistic copy();

    
    public abstract void clear();

    
    public abstract double getResult();

    
    public abstract void increment(final double d);

    
    public void incrementAll(double[] values) throws MathIllegalArgumentException {
        if (values == null) {
            throw new NullArgumentException(LocalizedFormats.INPUT_ARRAY);
        }
        incrementAll(values, 0, values.length);
    }

    
    public void incrementAll(double[] values, int begin, int length) throws MathIllegalArgumentException {
        if (test(values, begin, length)) {
            int k = begin + length;
            for (int i = begin; i < k; i++) {
                increment(values[i]);
            }
        }
    }

    
    @Override
    public boolean equals(Object object) {
        if (object == this ) {
            return true;
        }
       if (object instanceof AbstractStorelessUnivariateStatistic == false) {
            return false;
        }
        AbstractStorelessUnivariateStatistic stat = (AbstractStorelessUnivariateStatistic) object;
        return Precision.equalsIncludingNaN(stat.getResult(), this.getResult()) &&
               Precision.equalsIncludingNaN(stat.getN(), this.getN());
    }

    
    @Override
    public int hashCode() {
        return 31* (31 + MathUtils.hash(getResult())) + MathUtils.hash(getN());
    }

}
