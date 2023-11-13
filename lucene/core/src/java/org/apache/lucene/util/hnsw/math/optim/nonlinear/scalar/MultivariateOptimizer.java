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
package org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar;

import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.optim.BaseMultivariateOptimizer;
import org.apache.lucene.util.hnsw.math.optim.OptimizationData;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optim.PointValuePair;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;


public abstract class MultivariateOptimizer
    extends BaseMultivariateOptimizer<PointValuePair> {
    
    private MultivariateFunction function;
    
    private GoalType goal;

    
    protected MultivariateOptimizer(ConvergenceChecker<PointValuePair> checker) {
        super(checker);
    }

    
    @Override
    public PointValuePair optimize(OptimizationData... optData)
        throws TooManyEvaluationsException {
        // Set up base class and perform computation.
        return super.optimize(optData);
    }

    
    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        // Allow base class to register its own data.
        super.parseOptimizationData(optData);

        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof GoalType) {
                goal = (GoalType) data;
                continue;
            }
            if (data instanceof ObjectiveFunction) {
                function = ((ObjectiveFunction) data).getObjectiveFunction();
                continue;
            }
        }
    }

    
    public GoalType getGoalType() {
        return goal;
    }

    
    public double computeObjectiveValue(double[] params) {
        super.incrementEvaluationCount();
        return function.value(params);
    }
}
