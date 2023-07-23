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
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.stat.descriptive.StorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.stat.descriptive.summary.SumOfLogs;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class GeometricMean extends AbstractStorelessUnivariateStatistic implements Serializable {

    
    private static final long serialVersionUID = -8178734905303459453L;

    
    private StorelessUnivariateStatistic sumOfLogs;

    
    public GeometricMean() {
        sumOfLogs = new SumOfLogs();
    }

    
    public GeometricMean(GeometricMean original) throws NullArgumentException {
        super();
        copy(original, this);
    }

    
    public GeometricMean(SumOfLogs sumOfLogs) {
        this.sumOfLogs = sumOfLogs;
    }

    
    @Override
    public GeometricMean copy() {
        GeometricMean result = new GeometricMean();
        // no try-catch or advertised exception because args guaranteed non-null
        copy(this, result);
        return result;
    }

    
    @Override
    public void increment(final double d) {
        sumOfLogs.increment(d);
    }

    
    @Override
    public double getResult() {
        if (sumOfLogs.getN() > 0) {
            return FastMath.exp(sumOfLogs.getResult() / sumOfLogs.getN());
        } else {
            return Double.NaN;
        }
    }

    
    @Override
    public void clear() {
        sumOfLogs.clear();
    }

    
    @Override
    public double evaluate(
        final double[] values, final int begin, final int length)
    throws MathIllegalArgumentException {
        return FastMath.exp(
            sumOfLogs.evaluate(values, begin, length) / length);
    }

    
    public long getN() {
        return sumOfLogs.getN();
    }

    
    public void setSumLogImpl(StorelessUnivariateStatistic sumLogImpl)
    throws MathIllegalStateException {
        checkEmpty();
        this.sumOfLogs = sumLogImpl;
    }

    
    public StorelessUnivariateStatistic getSumLogImpl() {
        return sumOfLogs;
    }

    
    public static void copy(GeometricMean source, GeometricMean dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.sumOfLogs = source.sumOfLogs.copy();
    }


    
    private void checkEmpty() throws MathIllegalStateException {
        if (getN() > 0) {
            throw new MathIllegalStateException(
                    LocalizedFormats.VALUES_ADDED_BEFORE_CONFIGURING_STATISTIC,
                    getN());
        }
    }

}
