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
package org.apache.lucene.util.hnsw.math.analysis.integration;

import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class SimpsonIntegrator extends BaseAbstractUnivariateIntegrator {

    
    public static final int SIMPSON_MAX_ITERATIONS_COUNT = 64;

    
    public SimpsonIntegrator(final double relativeAccuracy,
                             final double absoluteAccuracy,
                             final int minimalIterationCount,
                             final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException, NumberIsTooLargeException {
        super(relativeAccuracy, absoluteAccuracy, minimalIterationCount, maximalIterationCount);
        if (maximalIterationCount > SIMPSON_MAX_ITERATIONS_COUNT) {
            throw new NumberIsTooLargeException(maximalIterationCount,
                                                SIMPSON_MAX_ITERATIONS_COUNT, false);
        }
    }

    
    public SimpsonIntegrator(final int minimalIterationCount,
                             final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException, NumberIsTooLargeException {
        super(minimalIterationCount, maximalIterationCount);
        if (maximalIterationCount > SIMPSON_MAX_ITERATIONS_COUNT) {
            throw new NumberIsTooLargeException(maximalIterationCount,
                                                SIMPSON_MAX_ITERATIONS_COUNT, false);
        }
    }

    
    public SimpsonIntegrator() {
        super(DEFAULT_MIN_ITERATIONS_COUNT, SIMPSON_MAX_ITERATIONS_COUNT);
    }

    
    @Override
    protected double doIntegrate()
        throws TooManyEvaluationsException, MaxCountExceededException {

        TrapezoidIntegrator qtrap = new TrapezoidIntegrator();
        if (getMinimalIterationCount() == 1) {
            return (4 * qtrap.stage(this, 1) - qtrap.stage(this, 0)) / 3.0;
        }

        // Simpson's rule requires at least two trapezoid stages.
        double olds = 0;
        double oldt = qtrap.stage(this, 0);
        while (true) {
            final double t = qtrap.stage(this, getIterations());
            incrementCount();
            final double s = (4 * t - oldt) / 3.0;
            if (getIterations() >= getMinimalIterationCount()) {
                final double delta = FastMath.abs(s - olds);
                final double rLimit =
                    getRelativeAccuracy() * (FastMath.abs(olds) + FastMath.abs(s)) * 0.5;
                if ((delta <= rLimit) || (delta <= getAbsoluteAccuracy())) {
                    return s;
                }
            }
            olds = s;
            oldt = t;
        }

    }

}
