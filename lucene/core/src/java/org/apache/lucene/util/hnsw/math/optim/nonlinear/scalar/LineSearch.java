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

import org.apache.lucene.util.hnsw.math.optim.univariate.UnivariateOptimizer;
import org.apache.lucene.util.hnsw.math.optim.univariate.BrentOptimizer;
import org.apache.lucene.util.hnsw.math.optim.univariate.BracketFinder;
import org.apache.lucene.util.hnsw.math.optim.univariate.UnivariatePointValuePair;
import org.apache.lucene.util.hnsw.math.optim.univariate.SimpleUnivariateValueChecker;
import org.apache.lucene.util.hnsw.math.optim.univariate.SearchInterval;
import org.apache.lucene.util.hnsw.math.optim.univariate.UnivariateObjectiveFunction;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.optim.MaxEval;


public class LineSearch {
    
    private static final double REL_TOL_UNUSED = 1e-15;
    
    private static final double ABS_TOL_UNUSED = Double.MIN_VALUE;
    
    private final UnivariateOptimizer lineOptimizer;
    
    private final BracketFinder bracket = new BracketFinder();
    
    private final double initialBracketingRange;
    
    private final MultivariateOptimizer mainOptimizer;

    
    public LineSearch(MultivariateOptimizer optimizer,
                      double relativeTolerance,
                      double absoluteTolerance,
                      double initialBracketingRange) {
        mainOptimizer = optimizer;
        lineOptimizer = new BrentOptimizer(REL_TOL_UNUSED,
                                           ABS_TOL_UNUSED,
                                           new SimpleUnivariateValueChecker(relativeTolerance,
                                                                            absoluteTolerance));
        this.initialBracketingRange = initialBracketingRange;
    }

    
    public UnivariatePointValuePair search(final double[] startPoint,
                                           final double[] direction) {
        final int n = startPoint.length;
        final UnivariateFunction f = new UnivariateFunction() {
                
                public double value(double alpha) {
                    final double[] x = new double[n];
                    for (int i = 0; i < n; i++) {
                        x[i] = startPoint[i] + alpha * direction[i];
                    }
                    final double obj = mainOptimizer.computeObjectiveValue(x);
                    return obj;
                }
            };

        final GoalType goal = mainOptimizer.getGoalType();
        bracket.search(f, goal, 0, initialBracketingRange);
        // Passing "MAX_VALUE" as a dummy value because it is the enclosing
        // class that counts the number of evaluations (and will eventually
        // generate the exception).
        return lineOptimizer.optimize(new MaxEval(Integer.MAX_VALUE),
                                      new UnivariateObjectiveFunction(f),
                                      goal,
                                      new SearchInterval(bracket.getLo(),
                                                         bracket.getHi(),
                                                         bracket.getMid()));
    }
}
