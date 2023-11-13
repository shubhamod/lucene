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

package org.apache.lucene.util.hnsw.math.optimization.general;

import org.apache.lucene.util.hnsw.math.analysis.MultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.GradientFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.MultivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optimization.GoalType;
import org.apache.lucene.util.hnsw.math.optimization.OptimizationData;
import org.apache.lucene.util.hnsw.math.optimization.InitialGuess;
import org.apache.lucene.util.hnsw.math.optimization.PointValuePair;
import org.apache.lucene.util.hnsw.math.optimization.direct.BaseAbstractMultivariateOptimizer;


@Deprecated
public abstract class AbstractDifferentiableOptimizer
    extends BaseAbstractMultivariateOptimizer<MultivariateDifferentiableFunction> {
    
    private MultivariateVectorFunction gradient;

    
    protected AbstractDifferentiableOptimizer(ConvergenceChecker<PointValuePair> checker) {
        super(checker);
    }

    
    protected double[] computeObjectiveGradient(final double[] evaluationPoint) {
        return gradient.value(evaluationPoint);
    }

    
    @Override@Deprecated
    protected PointValuePair optimizeInternal(final int maxEval,
                                              final MultivariateDifferentiableFunction f,
                                              final GoalType goalType,
                                              final double[] startPoint) {
        return optimizeInternal(maxEval, f, goalType, new InitialGuess(startPoint));
    }

    
    @Override
    protected PointValuePair optimizeInternal(final int maxEval,
                                              final MultivariateDifferentiableFunction f,
                                              final GoalType goalType,
                                              final OptimizationData... optData) {
        // Store optimization problem characteristics.
        gradient = new GradientFunction(f);

        // Perform optimization.
        return super.optimizeInternal(maxEval, f, goalType, optData);
    }
}
