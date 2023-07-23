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

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.integration.gauss.GaussIntegratorFactory;
import org.apache.lucene.util.hnsw.math.analysis.integration.gauss.GaussIntegrator;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class IterativeLegendreGaussIntegrator
    extends BaseAbstractUnivariateIntegrator {
    
    private static final GaussIntegratorFactory FACTORY
        = new GaussIntegratorFactory();
    
    private final int numberOfPoints;

    
    public IterativeLegendreGaussIntegrator(final int n,
                                            final double relativeAccuracy,
                                            final double absoluteAccuracy,
                                            final int minimalIterationCount,
                                            final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException {
        super(relativeAccuracy, absoluteAccuracy, minimalIterationCount, maximalIterationCount);
        if (n <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_POINTS, n);
        }
       numberOfPoints = n;
    }

    
    public IterativeLegendreGaussIntegrator(final int n,
                                            final double relativeAccuracy,
                                            final double absoluteAccuracy)
        throws NotStrictlyPositiveException {
        this(n, relativeAccuracy, absoluteAccuracy,
             DEFAULT_MIN_ITERATIONS_COUNT, DEFAULT_MAX_ITERATIONS_COUNT);
    }

    
    public IterativeLegendreGaussIntegrator(final int n,
                                            final int minimalIterationCount,
                                            final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException {
        this(n, DEFAULT_RELATIVE_ACCURACY, DEFAULT_ABSOLUTE_ACCURACY,
             minimalIterationCount, maximalIterationCount);
    }

    
    @Override
    protected double doIntegrate()
        throws MathIllegalArgumentException, TooManyEvaluationsException, MaxCountExceededException {
        // Compute first estimate with a single step.
        double oldt = stage(1);

        int n = 2;
        while (true) {
            // Improve integral with a larger number of steps.
            final double t = stage(n);

            // Estimate the error.
            final double delta = FastMath.abs(t - oldt);
            final double limit =
                FastMath.max(getAbsoluteAccuracy(),
                             getRelativeAccuracy() * (FastMath.abs(oldt) + FastMath.abs(t)) * 0.5);

            // check convergence
            if (getIterations() + 1 >= getMinimalIterationCount() &&
                delta <= limit) {
                return t;
            }

            // Prepare next iteration.
            final double ratio = FastMath.min(4, FastMath.pow(delta / limit, 0.5 / numberOfPoints));
            n = FastMath.max((int) (ratio * n), n + 1);
            oldt = t;
            incrementCount();
        }
    }

    
    private double stage(final int n)
        throws TooManyEvaluationsException {
        // Function to be integrated is stored in the base class.
        final UnivariateFunction f = new UnivariateFunction() {
                
                public double value(double x)
                    throws MathIllegalArgumentException, TooManyEvaluationsException {
                    return computeObjectiveValue(x);
                }
            };

        final double min = getMin();
        final double max = getMax();
        final double step = (max - min) / n;

        double sum = 0;
        for (int i = 0; i < n; i++) {
            // Integrate over each sub-interval [a, b].
            final double a = min + i * step;
            final double b = a + step;
            final GaussIntegrator g = FACTORY.legendreHighPrecision(numberOfPoints, a, b);
            sum += g.integrate(f);
        }

        return sum;
    }
}
