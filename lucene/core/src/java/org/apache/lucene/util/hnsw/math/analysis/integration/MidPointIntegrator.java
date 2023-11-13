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


public class MidPointIntegrator extends BaseAbstractUnivariateIntegrator {

    
    public static final int MIDPOINT_MAX_ITERATIONS_COUNT = 64;

    
    public MidPointIntegrator(final double relativeAccuracy,
                              final double absoluteAccuracy,
                              final int minimalIterationCount,
                              final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException, NumberIsTooLargeException {
        super(relativeAccuracy, absoluteAccuracy, minimalIterationCount, maximalIterationCount);
        if (maximalIterationCount > MIDPOINT_MAX_ITERATIONS_COUNT) {
            throw new NumberIsTooLargeException(maximalIterationCount,
                                                MIDPOINT_MAX_ITERATIONS_COUNT, false);
        }
    }

    
    public MidPointIntegrator(final int minimalIterationCount,
                              final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException, NumberIsTooLargeException {
        super(minimalIterationCount, maximalIterationCount);
        if (maximalIterationCount > MIDPOINT_MAX_ITERATIONS_COUNT) {
            throw new NumberIsTooLargeException(maximalIterationCount,
                                                MIDPOINT_MAX_ITERATIONS_COUNT, false);
        }
    }

    
    public MidPointIntegrator() {
        super(DEFAULT_MIN_ITERATIONS_COUNT, MIDPOINT_MAX_ITERATIONS_COUNT);
    }

    
    private double stage(final int n,
                         double previousStageResult,
                         double min,
                         double diffMaxMin)
        throws TooManyEvaluationsException {

        // number of new points in this stage
        final long np = 1L << (n - 1);
        double sum = 0;

        // spacing between adjacent new points
        final double spacing = diffMaxMin / np;

        // the first new point
        double x = min + 0.5 * spacing;
        for (long i = 0; i < np; i++) {
            sum += computeObjectiveValue(x);
            x += spacing;
        }
        // add the new sum to previously calculated result
        return 0.5 * (previousStageResult + sum * spacing);
    }


    
    @Override
    protected double doIntegrate()
        throws MathIllegalArgumentException, TooManyEvaluationsException, MaxCountExceededException {

        final double min = getMin();
        final double diff = getMax() - min;
        final double midPoint = min + 0.5 * diff;

        double oldt = diff * computeObjectiveValue(midPoint);

        while (true) {
            incrementCount();
            final int i = getIterations();
            final double t = stage(i, oldt, min, diff);
            if (i >= getMinimalIterationCount()) {
                final double delta = FastMath.abs(t - oldt);
                final double rLimit =
                        getRelativeAccuracy() * (FastMath.abs(oldt) + FastMath.abs(t)) * 0.5;
                if ((delta <= rLimit) || (delta <= getAbsoluteAccuracy())) {
                    return t;
                }
            }
            oldt = t;
        }

    }

}
