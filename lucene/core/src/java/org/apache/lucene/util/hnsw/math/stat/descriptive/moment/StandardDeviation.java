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
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class StandardDeviation extends AbstractStorelessUnivariateStatistic
    implements Serializable {

    
    private static final long serialVersionUID = 5728716329662425188L;

    
    private Variance variance = null;

    
    public StandardDeviation() {
        variance = new Variance();
    }

    
    public StandardDeviation(final SecondMoment m2) {
        variance = new Variance(m2);
    }

    
    public StandardDeviation(StandardDeviation original) throws NullArgumentException {
        copy(original, this);
    }

    
    public StandardDeviation(boolean isBiasCorrected) {
        variance = new Variance(isBiasCorrected);
    }

    
    public StandardDeviation(boolean isBiasCorrected, SecondMoment m2) {
        variance = new Variance(isBiasCorrected, m2);
    }

    
    @Override
    public void increment(final double d) {
        variance.increment(d);
    }

    
    public long getN() {
        return variance.getN();
    }

    
    @Override
    public double getResult() {
        return FastMath.sqrt(variance.getResult());
    }

    
    @Override
    public void clear() {
        variance.clear();
    }

    
    @Override
    public double evaluate(final double[] values) throws MathIllegalArgumentException  {
        return FastMath.sqrt(variance.evaluate(values));
    }

    
    @Override
    public double evaluate(final double[] values, final int begin, final int length)
    throws MathIllegalArgumentException  {
       return FastMath.sqrt(variance.evaluate(values, begin, length));
    }

    
    public double evaluate(final double[] values, final double mean,
            final int begin, final int length) throws MathIllegalArgumentException  {
        return FastMath.sqrt(variance.evaluate(values, mean, begin, length));
    }

    
    public double evaluate(final double[] values, final double mean)
    throws MathIllegalArgumentException  {
        return FastMath.sqrt(variance.evaluate(values, mean));
    }

    
    public boolean isBiasCorrected() {
        return variance.isBiasCorrected();
    }

    
    public void setBiasCorrected(boolean isBiasCorrected) {
        variance.setBiasCorrected(isBiasCorrected);
    }

    
    @Override
    public StandardDeviation copy() {
        StandardDeviation result = new StandardDeviation();
        // No try-catch or advertised exception because args are guaranteed non-null
        copy(this, result);
        return result;
    }


    
    public static void copy(StandardDeviation source, StandardDeviation dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.variance = source.variance.copy();
    }

}
