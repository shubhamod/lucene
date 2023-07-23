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

import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.optimization.BaseMultivariateOptimizer;
import org.apache.lucene.util.hnsw.math.optimization.BaseMultivariateSimpleBoundsOptimizer;
import org.apache.lucene.util.hnsw.math.optimization.GoalType;
import org.apache.lucene.util.hnsw.math.optimization.InitialGuess;
import org.apache.lucene.util.hnsw.math.optimization.SimpleBounds;
import org.apache.lucene.util.hnsw.math.optimization.PointValuePair;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;


@Deprecated
public abstract class BaseAbstractMultivariateSimpleBoundsOptimizer<FUNC extends MultivariateFunction>
    extends BaseAbstractMultivariateOptimizer<FUNC>
    implements BaseMultivariateOptimizer<FUNC>,
               BaseMultivariateSimpleBoundsOptimizer<FUNC> {
    
    @Deprecated
    protected BaseAbstractMultivariateSimpleBoundsOptimizer() {}

    
    protected BaseAbstractMultivariateSimpleBoundsOptimizer(ConvergenceChecker<PointValuePair> checker) {
        super(checker);
    }

    
    @Override
    public PointValuePair optimize(int maxEval, FUNC f, GoalType goalType,
                                   double[] startPoint) {
        return super.optimizeInternal(maxEval, f, goalType,
                                      new InitialGuess(startPoint));
    }

    
    public PointValuePair optimize(int maxEval, FUNC f, GoalType goalType,
                                   double[] startPoint,
                                   double[] lower, double[] upper) {
        return super.optimizeInternal(maxEval, f, goalType,
                                      new InitialGuess(startPoint),
                                      new SimpleBounds(lower, upper));
    }
}
