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
import org.apache.lucene.util.hnsw.math.analysis.solvers.UnivariateSolverUtils;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.util.IntegerSequence;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public abstract class BaseAbstractUnivariateIntegrator implements UnivariateIntegrator {

    
    public static final double DEFAULT_ABSOLUTE_ACCURACY = 1.0e-15;

    
    public static final double DEFAULT_RELATIVE_ACCURACY = 1.0e-6;

    
    public static final int DEFAULT_MIN_ITERATIONS_COUNT = 3;

    
    public static final int DEFAULT_MAX_ITERATIONS_COUNT = Integer.MAX_VALUE;

    
    @Deprecated
    protected org.apache.lucene.util.hnsw.math.util.Incrementor iterations;

    
    private IntegerSequence.Incrementor count;

    
    private final double absoluteAccuracy;

    
    private final double relativeAccuracy;

    
    private final int minimalIterationCount;

    
    private IntegerSequence.Incrementor evaluations;

    
    private UnivariateFunction function;

    
    private double min;

    
    private double max;

    
    protected BaseAbstractUnivariateIntegrator(final double relativeAccuracy,
                                               final double absoluteAccuracy,
                                               final int minimalIterationCount,
                                               final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException {

        // accuracy settings
        this.relativeAccuracy      = relativeAccuracy;
        this.absoluteAccuracy      = absoluteAccuracy;

        // iterations count settings
        if (minimalIterationCount <= 0) {
            throw new NotStrictlyPositiveException(minimalIterationCount);
        }
        if (maximalIterationCount <= minimalIterationCount) {
            throw new NumberIsTooSmallException(maximalIterationCount, minimalIterationCount, false);
        }
        this.minimalIterationCount = minimalIterationCount;
        this.count                 = IntegerSequence.Incrementor.create().withMaximalCount(maximalIterationCount);

        @SuppressWarnings("deprecation")
        org.apache.lucene.util.hnsw.math.util.Incrementor wrapped =
                        org.apache.lucene.util.hnsw.math.util.Incrementor.wrap(count);
        this.iterations = wrapped;

        // prepare evaluations counter, but do not set it yet
        evaluations = IntegerSequence.Incrementor.create();

    }

    
    protected BaseAbstractUnivariateIntegrator(final double relativeAccuracy,
                                           final double absoluteAccuracy) {
        this(relativeAccuracy, absoluteAccuracy,
             DEFAULT_MIN_ITERATIONS_COUNT, DEFAULT_MAX_ITERATIONS_COUNT);
    }

    
    protected BaseAbstractUnivariateIntegrator(final int minimalIterationCount,
                                           final int maximalIterationCount)
        throws NotStrictlyPositiveException, NumberIsTooSmallException {
        this(DEFAULT_RELATIVE_ACCURACY, DEFAULT_ABSOLUTE_ACCURACY,
             minimalIterationCount, maximalIterationCount);
    }

    
    public double getRelativeAccuracy() {
        return relativeAccuracy;
    }

    
    public double getAbsoluteAccuracy() {
        return absoluteAccuracy;
    }

    
    public int getMinimalIterationCount() {
        return minimalIterationCount;
    }

    
    public int getMaximalIterationCount() {
        return count.getMaximalCount();
    }

    
    public int getEvaluations() {
        return evaluations.getCount();
    }

    
    public int getIterations() {
        return count.getCount();
    }

    
    protected void incrementCount() throws MaxCountExceededException {
        count.increment();
    }

    
    protected double getMin() {
        return min;
    }
    
    protected double getMax() {
        return max;
    }

    
    protected double computeObjectiveValue(final double point)
        throws TooManyEvaluationsException {
        try {
            evaluations.increment();
        } catch (MaxCountExceededException e) {
            throw new TooManyEvaluationsException(e.getMax());
        }
        return function.value(point);
    }

    
    protected void setup(final int maxEval,
                         final UnivariateFunction f,
                         final double lower, final double upper)
        throws NullArgumentException, MathIllegalArgumentException {

        // Checks.
        MathUtils.checkNotNull(f);
        UnivariateSolverUtils.verifyInterval(lower, upper);

        // Reset.
        min = lower;
        max = upper;
        function = f;
        evaluations = evaluations.withMaximalCount(maxEval).withStart(0);
        count       = count.withStart(0);

    }

    
    public double integrate(final int maxEval, final UnivariateFunction f,
                            final double lower, final double upper)
        throws TooManyEvaluationsException, MaxCountExceededException,
               MathIllegalArgumentException, NullArgumentException {

        // Initialization.
        setup(maxEval, f, lower, upper);

        // Perform computation.
        return doIntegrate();

    }

    
    protected abstract double doIntegrate()
        throws TooManyEvaluationsException, MaxCountExceededException;

}
