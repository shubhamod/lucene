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

package org.apache.lucene.util.hnsw.math.optimization.univariate;

import org.apache.lucene.util.hnsw.math.util.Incrementor;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.optimization.GoalType;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;


@Deprecated
public abstract class BaseAbstractUnivariateOptimizer
    implements UnivariateOptimizer {
    
    private final ConvergenceChecker<UnivariatePointValuePair> checker;
    
    private final Incrementor evaluations = new Incrementor();
    
    private GoalType goal;
    
    private double searchMin;
    
    private double searchMax;
    
    private double searchStart;
    
    private UnivariateFunction function;

    
    protected BaseAbstractUnivariateOptimizer(ConvergenceChecker<UnivariatePointValuePair> checker) {
        this.checker = checker;
    }

    
    public int getMaxEvaluations() {
        return evaluations.getMaximalCount();
    }

    
    public int getEvaluations() {
        return evaluations.getCount();
    }

    
    public GoalType getGoalType() {
        return goal;
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

    
    protected double computeObjectiveValue(double point) {
        try {
            evaluations.incrementCount();
        } catch (MaxCountExceededException e) {
            throw new TooManyEvaluationsException(e.getMax());
        }
        return function.value(point);
    }

    
    public UnivariatePointValuePair optimize(int maxEval, UnivariateFunction f,
                                             GoalType goalType,
                                             double min, double max,
                                             double startValue) {
        // Checks.
        if (f == null) {
            throw new NullArgumentException();
        }
        if (goalType == null) {
            throw new NullArgumentException();
        }

        // Reset.
        searchMin = min;
        searchMax = max;
        searchStart = startValue;
        goal = goalType;
        function = f;
        evaluations.setMaximalCount(maxEval);
        evaluations.resetCount();

        // Perform computation.
        return doOptimize();
    }

    
    public UnivariatePointValuePair optimize(int maxEval,
                                             UnivariateFunction f,
                                             GoalType goalType,
                                             double min, double max){
        return optimize(maxEval, f, goalType, min, max, min + 0.5 * (max - min));
    }

    
    public ConvergenceChecker<UnivariatePointValuePair> getConvergenceChecker() {
        return checker;
    }

    
    protected abstract UnivariatePointValuePair doOptimize();
}
