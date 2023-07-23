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
package org.apache.lucene.util.hnsw.math.analysis.interpolation;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;


public class UnivariatePeriodicInterpolator
    implements UnivariateInterpolator {
    
    public static final int DEFAULT_EXTEND = 5;
    
    private final UnivariateInterpolator interpolator;
    
    private final double period;
    
    private final int extend;

    
    public UnivariatePeriodicInterpolator(UnivariateInterpolator interpolator,
                                          double period,
                                          int extend) {
        this.interpolator = interpolator;
        this.period = period;
        this.extend = extend;
    }

    
    public UnivariatePeriodicInterpolator(UnivariateInterpolator interpolator,
                                          double period) {
        this(interpolator, period, DEFAULT_EXTEND);
    }

    
    public UnivariateFunction interpolate(double[] xval,
                                          double[] yval)
        throws NumberIsTooSmallException, NonMonotonicSequenceException {
        if (xval.length < extend) {
            throw new NumberIsTooSmallException(xval.length, extend, true);
        }

        MathArrays.checkOrder(xval);
        final double offset = xval[0];

        final int len = xval.length + extend * 2;
        final double[] x = new double[len];
        final double[] y = new double[len];
        for (int i = 0; i < xval.length; i++) {
            final int index = i + extend;
            x[index] = MathUtils.reduce(xval[i], period, offset);
            y[index] = yval[i];
        }

        // Wrap to enable interpolation at the boundaries.
        for (int i = 0; i < extend; i++) {
            int index = xval.length - extend + i;
            x[i] = MathUtils.reduce(xval[index], period, offset) - period;
            y[i] = yval[index];

            index = len - extend + i;
            x[index] = MathUtils.reduce(xval[i], period, offset) + period;
            y[index] = yval[i];
        }

        MathArrays.sortInPlace(x, y);

        final UnivariateFunction f = interpolator.interpolate(x, y);
        return new UnivariateFunction() {
            
            public double value(final double x) throws MathIllegalArgumentException {
                return f.value(MathUtils.reduce(x, period, offset));
            }
        };
    }
}
