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

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class TrapezoidIntegrator extends BaseAbstractUnivariateIntegrator {

    
    public static final int TRAPEZOID_MAX_ITERATIONS_COUNT = 64;

    
    private double s;

    
    public TrapezoidIntegrator(final double relativeAccuracy,
                               final double absoluteAccuracy,
                               final int minimalIterationCount,
                               final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException, NumberIsTooLargeException {
        super(relativeAccuracy, absoluteAccuracy, minimalIterationCount, maximalIterationCount);
        if (maximalIterationCount > TRAPEZOID_MAX_ITERATIONS_COUNT) {
            throw new NumberIsTooLargeException(maximalIterationCount,
                                                TRAPEZOID_MAX_ITERATIONS_COUNT, false);
        }
    }

    
    public TrapezoidIntegrator(final int minimalIterationCount,
                               final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException, NumberIsTooLargeException {
        super(minimalIterationCount, maximalIterationCount);
        if (maximalIterationCount > TRAPEZOID_MAX_ITERATIONS_COUNT) {
            throw new NumberIsTooLargeException(maximalIterationCount,
                                                TRAPEZOID_MAX_ITERATIONS_COUNT, false);
        }
    }

    
    public TrapezoidIntegrator() {
        super(DEFAULT_MIN_ITERATIONS_COUNT, TRAPEZOID_MAX_ITERATIONS_COUNT);
    }

    
    double stage(final BaseAbstractUnivariateIntegrator baseIntegrator, final int n)
        throws TooManyEvaluationsException {

        if (n == 0) {
            final double max = baseIntegrator.getMax();
            final double min = baseIntegrator.getMin();
            s = 0.5 * (max - min) *
                      (baseIntegrator.computeObjectiveValue(min) +
                       baseIntegrator.computeObjectiveValue(max));
            return s;
        } else {
            final long np = 1L << (n-1);           // number of new points in this stage
            double sum = 0;
            final double max = baseIntegrator.getMax();
            final double min = baseIntegrator.getMin();
            // spacing between adjacent new points
            final double spacing = (max - min) / np;
            double x = min + 0.5 * spacing;    // the first new point
            for (long i = 0; i < np; i++) {
                sum += baseIntegrator.computeObjectiveValue(x);
                x += spacing;
            }
            // add the new sum to previously calculated result
            s = 0.5 * (s + sum * spacing);
            return s;
        }
    }

    
    @Override
    protected double doIntegrate()
        throws MathIllegalArgumentException, TooManyEvaluationsException, MaxCountExceededException {

        double oldt = stage(this, 0);
        incrementCount();
        while (true) {
            final int i = getIterations();
            final double t = stage(this, i);
            if (i >= getMinimalIterationCount()) {
                final double delta = FastMath.abs(t - oldt);
                final double rLimit =
                    getRelativeAccuracy() * (FastMath.abs(oldt) + FastMath.abs(t)) * 0.5;
                if ((delta <= rLimit) || (delta <= getAbsoluteAccuracy())) {
                    return t;
                }
            }
            oldt = t;
            incrementCount();
        }

    }

}
