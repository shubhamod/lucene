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

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class SynchronizedDescriptiveStatistics extends DescriptiveStatistics {

    
    private static final long serialVersionUID = 1L;

    
    public SynchronizedDescriptiveStatistics() {
        // no try-catch or advertized IAE because arg is valid
        this(INFINITE_WINDOW);
    }

    
    public SynchronizedDescriptiveStatistics(int window) throws MathIllegalArgumentException {
        super(window);
    }

    
    public SynchronizedDescriptiveStatistics(SynchronizedDescriptiveStatistics original)
    throws NullArgumentException {
        copy(original, this);
    }

    
    @Override
    public synchronized void addValue(double v) {
        super.addValue(v);
    }

    
    @Override
    public synchronized double apply(UnivariateStatistic stat) {
        return super.apply(stat);
    }

    
    @Override
    public synchronized void clear() {
        super.clear();
    }

    
    @Override
    public synchronized double getElement(int index) {
        return super.getElement(index);
    }

    
    @Override
    public synchronized long getN() {
        return super.getN();
    }

    
    @Override
    public synchronized double getStandardDeviation() {
        return super.getStandardDeviation();
    }

    
    @Override
    public synchronized double getQuadraticMean() {
        return super.getQuadraticMean();
    }

    
    @Override
    public synchronized double[] getValues() {
        return super.getValues();
    }

    
    @Override
    public synchronized int getWindowSize() {
        return super.getWindowSize();
    }

    
    @Override
    public synchronized void setWindowSize(int windowSize) throws MathIllegalArgumentException {
        super.setWindowSize(windowSize);
    }

    
    @Override
    public synchronized String toString() {
        return super.toString();
    }

    
    @Override
    public synchronized SynchronizedDescriptiveStatistics copy() {
        SynchronizedDescriptiveStatistics result =
            new SynchronizedDescriptiveStatistics();
        // No try-catch or advertised exception because arguments are guaranteed non-null
        copy(this, result);
        return result;
    }

    
    public static void copy(SynchronizedDescriptiveStatistics source,
                            SynchronizedDescriptiveStatistics dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        synchronized (source) {
            synchronized (dest) {
                DescriptiveStatistics.copy(source, dest);
            }
        }
    }
}
