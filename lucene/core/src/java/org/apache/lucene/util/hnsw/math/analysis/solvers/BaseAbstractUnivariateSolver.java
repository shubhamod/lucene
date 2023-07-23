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

package org.apache.lucene.util.hnsw.math.analysis.solvers;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.util.IntegerSequence;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public abstract class BaseAbstractUnivariateSolver<FUNC extends UnivariateFunction>
    implements BaseUnivariateSolver<FUNC> {
    
    private static final double DEFAULT_RELATIVE_ACCURACY = 1e-14;
    
    private static final double DEFAULT_FUNCTION_VALUE_ACCURACY = 1e-15;
    
    private final double functionValueAccuracy;
    
    private final double absoluteAccuracy;
    
    private final double relativeAccuracy;
    
    private IntegerSequence.Incrementor evaluations;
    
    private double searchMin;
    
    private double searchMax;
    
    private double searchStart;
    
    private FUNC function;

    
    protected BaseAbstractUnivariateSolver(final double absoluteAccuracy) {
        this(DEFAULT_RELATIVE_ACCURACY,
             absoluteAccuracy,
             DEFAULT_FUNCTION_VALUE_ACCURACY);
    }

    
    protected BaseAbstractUnivariateSolver(final double relativeAccuracy,
                                           final double absoluteAccuracy) {
        this(relativeAccuracy,
             absoluteAccuracy,
             DEFAULT_FUNCTION_VALUE_ACCURACY);
    }

    
    protected BaseAbstractUnivariateSolver(final double relativeAccuracy,
                                           final double absoluteAccuracy,
                                           final double functionValueAccuracy) {
        this.absoluteAccuracy      = absoluteAccuracy;
        this.relativeAccuracy      = relativeAccuracy;
        this.functionValueAccuracy = functionValueAccuracy;
        this.evaluations           = IntegerSequence.Incrementor.create();
    }

    
    public int getMaxEvaluations() {
        return evaluations.getMaximalCount();
    }
    
    public int getEvaluations() {
        return evaluations.getCount();
    }
    
    public double getMin() {
        return searchMin;
    }
    
    public double getMax() {
        return searchMax;
    }
    
    public double getStartValue() {
        return searchStart;
    }
    
    public double getAbsoluteAccuracy() {
        return absoluteAccuracy;
    }
    
    public double getRelativeAccuracy() {
        return relativeAccuracy;
    }
    
    public double getFunctionValueAccuracy() {
        return functionValueAccuracy;
    }

    
    protected double computeObjectiveValue(double point)
        throws TooManyEvaluationsException {
        incrementEvaluationCount();
        return function.value(point);
    }

    
    protected void setup(int maxEval,
                         FUNC f,
                         double min, double max,
                         double startValue)
        throws NullArgumentException {
        // Checks.
        MathUtils.checkNotNull(f);

        // Reset.
        searchMin = min;
        searchMax = max;
        searchStart = startValue;
        function = f;
        evaluations = evaluations.withMaximalCount(maxEval).withStart(0);
    }

    
    public double solve(int maxEval, FUNC f, double min, double max, double startValue)
        throws TooManyEvaluationsException,
               NoBracketingException {
        // Initialization.
        setup(maxEval, f, min, max, startValue);

        // Perform computation.
        return doSolve();
    }

    
    public double solve(int maxEval, FUNC f, double min, double max) {
        return solve(maxEval, f, min, max, min + 0.5 * (max - min));
    }

    
    public double solve(int maxEval, FUNC f, double startValue)
        throws TooManyEvaluationsException,
               NoBracketingException {
        return solve(maxEval, f, Double.NaN, Double.NaN, startValue);
    }

    
    protected abstract double doSolve()
        throws TooManyEvaluationsException, NoBracketingException;

    
    protected boolean isBracketing(final double lower,
                                   final double upper) {
        return UnivariateSolverUtils.isBracketing(function, lower, upper);
    }

    
    protected boolean isSequence(final double start,
                                 final double mid,
                                 final double end) {
        return UnivariateSolverUtils.isSequence(start, mid, end);
    }

    
    protected void verifyInterval(final double lower,
                                  final double upper)
        throws NumberIsTooLargeException {
        UnivariateSolverUtils.verifyInterval(lower, upper);
    }

    
    protected void verifySequence(final double lower,
                                  final double initial,
                                  final double upper)
        throws NumberIsTooLargeException {
        UnivariateSolverUtils.verifySequence(lower, initial, upper);
    }

    
    protected void verifyBracketing(final double lower,
                                    final double upper)
        throws NullArgumentException,
               NoBracketingException {
        UnivariateSolverUtils.verifyBracketing(function, lower, upper);
    }

    
    protected void incrementEvaluationCount()
        throws TooManyEvaluationsException {
        try {
            evaluations.increment();
        } catch (MaxCountExceededException e) {
            throw new TooManyEvaluationsException(e.getMax());
        }
    }
}
