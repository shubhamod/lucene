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

import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public abstract class AbstractUnivariateStatistic
    implements UnivariateStatistic {

    
    private double[] storedData;

    
    public void setData(final double[] values) {
        storedData = (values == null) ? null : values.clone();
    }

    
    public double[] getData() {
        return (storedData == null) ? null : storedData.clone();
    }

    
    protected double[] getDataRef() {
        return storedData;
    }

    
    public void setData(final double[] values, final int begin, final int length)
    throws MathIllegalArgumentException {
        if (values == null) {
            throw new NullArgumentException(LocalizedFormats.INPUT_ARRAY);
        }

        if (begin < 0) {
            throw new NotPositiveException(LocalizedFormats.START_POSITION, begin);
        }

        if (length < 0) {
            throw new NotPositiveException(LocalizedFormats.LENGTH, length);
        }

        if (begin + length > values.length) {
            throw new NumberIsTooLargeException(LocalizedFormats.SUBARRAY_ENDS_AFTER_ARRAY_END,
                                                begin + length, values.length, true);
        }
        storedData = new double[length];
        System.arraycopy(values, begin, storedData, 0, length);
    }

    
    public double evaluate() throws MathIllegalArgumentException {
        return evaluate(storedData);
    }

    
    public double evaluate(final double[] values) throws MathIllegalArgumentException {
        test(values, 0, 0);
        return evaluate(values, 0, values.length);
    }

    
    public abstract double evaluate(final double[] values, final int begin, final int length)
    throws MathIllegalArgumentException;

    
    public abstract UnivariateStatistic copy();

    
    protected boolean test(
        final double[] values,
        final int begin,
        final int length) throws MathIllegalArgumentException {
        return MathArrays.verifyValues(values, begin, length, false);
    }

    
    protected boolean test(final double[] values, final int begin,
            final int length, final boolean allowEmpty) throws MathIllegalArgumentException {
        return MathArrays.verifyValues(values, begin, length, allowEmpty);
    }

    
    protected boolean test(
        final double[] values,
        final double[] weights,
        final int begin,
        final int length) throws MathIllegalArgumentException {
        return MathArrays.verifyValues(values, weights, begin, length, false);
    }

    
    protected boolean test(final double[] values, final double[] weights,
            final int begin, final int length, final boolean allowEmpty) throws MathIllegalArgumentException {

        return MathArrays.verifyValues(values, weights, begin, length, allowEmpty);
    }
}

