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

package org.apache.lucene.util.hnsw.math.optimization.direct;

import org.apache.lucene.util.hnsw.math.util.Incrementor;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.optimization.BaseMultivariateOptimizer;
import org.apache.lucene.util.hnsw.math.optimization.OptimizationData;
import org.apache.lucene.util.hnsw.math.optimization.GoalType;
import org.apache.lucene.util.hnsw.math.optimization.InitialGuess;
import org.apache.lucene.util.hnsw.math.optimization.SimpleBounds;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optimization.PointValuePair;
import org.apache.lucene.util.hnsw.math.optimization.SimpleValueChecker;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;


@Deprecated
public abstract class BaseAbstractMultivariateOptimizer<FUNC extends MultivariateFunction>
    implements BaseMultivariateOptimizer<FUNC> {
    
    protected final Incrementor evaluations = new Incrementor();
    
    private ConvergenceChecker<PointValuePair> checker;
    
    private GoalType goal;
    
    private double[] start;
    
    private double[] lowerBound;
    
    private double[] upperBound;
    
    private MultivariateFunction function;

    
    @Deprecated
    protected BaseAbstractMultivariateOptimizer() {
        this(new SimpleValueChecker());
    }
    
    protected BaseAbstractMultivariateOptimizer(ConvergenceChecker<PointValuePair> checker) {
        this.checker = checker;
    }

    
    public int getMaxEvaluations() {
        return evaluations.getMaximalCount();
    }

    
    public int getEvaluations() {
        return evaluations.getCount();
    }

    
    public ConvergenceChecker<PointValuePair> getConvergenceChecker() {
        return checker;
    }

    
    protected double computeObjectiveValue(double[] point) {
        try {
            evaluations.incrementCount();
        } catch (MaxCountExceededException e) {
            throw new TooManyEvaluationsException(e.getMax());
        }
        return function.value(point);
    }

    
    @Deprecated
    public PointValuePair optimize(int maxEval, FUNC f, GoalType goalType,
                                   double[] startPoint) {
        return optimizeInternal(maxEval, f, goalType, new InitialGuess(startPoint));
    }

    
    public PointValuePair optimize(int maxEval,
                                   FUNC f,
                                   GoalType goalType,
                                   OptimizationData... optData) {
        return optimizeInternal(maxEval, f, goalType, optData);
    }

    
    @Deprecated
    protected PointValuePair optimizeInternal(int maxEval, FUNC f, GoalType goalType,
                                              double[] startPoint) {
        return optimizeInternal(maxEval, f, goalType, new InitialGuess(startPoint));
    }

    
    protected PointValuePair optimizeInternal(int maxEval,
                                              FUNC f,
                                              GoalType goalType,
                                              OptimizationData... optData)
        throws TooManyEvaluationsException {
        // Set internal state.
        evaluations.setMaximalCount(maxEval);
        evaluations.resetCount();
        function = f;
        goal = goalType;
        // Retrieve other settings.
        parseOptimizationData(optData);
        // Check input consistency.
        checkParameters();
        // Perform computation.
        return doOptimize();
    }

    
    private void parseOptimizationData(OptimizationData... optData) {
        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof InitialGuess) {
                start = ((InitialGuess) data).getInitialGuess();
                continue;
            }
            if (data instanceof SimpleBounds) {
                final SimpleBounds bounds = (SimpleBounds) data;
                lowerBound = bounds.getLower();
                upperBound = bounds.getUpper();
                continue;
            }
        }
    }

    
    public GoalType getGoalType() {
        return goal;
    }

    
    public double[] getStartPoint() {
        return start == null ? null : start.clone();
    }
    
    public double[] getLowerBound() {
        return lowerBound == null ? null : lowerBound.clone();
    }
    
    public double[] getUpperBound() {
        return upperBound == null ? null : upperBound.clone();
    }

    
    protected abstract PointValuePair doOptimize();

    
    private void checkParameters() {
        if (start != null) {
            final int dim = start.length;
            if (lowerBound != null) {
                if (lowerBound.length != dim) {
                    throw new DimensionMismatchException(lowerBound.length, dim);
                }
                for (int i = 0; i < dim; i++) {
                    final double v = start[i];
                    final double lo = lowerBound[i];
                    if (v < lo) {
                        throw new NumberIsTooSmallException(v, lo, true);
                    }
                }
            }
            if (upperBound != null) {
                if (upperBound.length != dim) {
                    throw new DimensionMismatchException(upperBound.length, dim);
                }
                for (int i = 0; i < dim; i++) {
                    final double v = start[i];
                    final double hi = upperBound[i];
                    if (v > hi) {
                        throw new NumberIsTooLargeException(v, hi, true);
                    }
                }
            }

            // If the bounds were not specified, the allowed interval is
            // assumed to be [-inf, +inf].
            if (lowerBound == null) {
                lowerBound = new double[dim];
                for (int i = 0; i < dim; i++) {
                    lowerBound[i] = Double.NEGATIVE_INFINITY;
                }
            }
            if (upperBound == null) {
                upperBound = new double[dim];
                for (int i = 0; i < dim; i++) {
                    upperBound[i] = Double.POSITIVE_INFINITY;
                }
            }
        }
    }
}
