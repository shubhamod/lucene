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
package org.apache.lucene.util.hnsw.math.optim.univariate;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.optim.BaseOptimizer;
import org.apache.lucene.util.hnsw.math.optim.OptimizationData;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.GoalType;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;


public abstract class UnivariateOptimizer
    extends BaseOptimizer<UnivariatePointValuePair> {
    
    private UnivariateFunction function;
    
    private GoalType goal;
    
    private double start;
    
    private double min;
    
    private double max;

    
    protected UnivariateOptimizer(ConvergenceChecker<UnivariatePointValuePair> checker) {
        super(checker);
    }

    
    @Override
    public UnivariatePointValuePair optimize(OptimizationData... optData)
        throws TooManyEvaluationsException {
        // Perform computation.
        return super.optimize(optData);
    }

    
    public GoalType getGoalType() {
        return goal;
    }

    
    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        // Allow base class to register its own data.
        super.parseOptimizationData(optData);

        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof SearchInterval) {
                final SearchInterval interval = (SearchInterval) data;
                min = interval.getMin();
                max = interval.getMax();
                start = interval.getStartValue();
                continue;
            }
            if (data instanceof UnivariateObjectiveFunction) {
                function = ((UnivariateObjectiveFunction) data).getObjectiveFunction();
                continue;
            }
            if (data instanceof GoalType) {
                goal = (GoalType) data;
                continue;
            }
        }
    }

    
    public double getStartValue() {
        return start;
    }
    
    public double getMin() {
        return min;
    }
    
    public double getMax() {
        return max;
    }

    
    protected double computeObjectiveValue(double x) {
        super.incrementEvaluationCount();
        return function.value(x);
    }
}
