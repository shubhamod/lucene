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

import java.util.Arrays;
import java.util.Comparator;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.optim.MaxEval;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.GoalType;
import org.apache.lucene.util.hnsw.math.optim.OptimizationData;


public class MultiStartUnivariateOptimizer
    extends UnivariateOptimizer {
    
    private final UnivariateOptimizer optimizer;
    
    private int totalEvaluations;
    
    private int starts;
    
    private RandomGenerator generator;
    
    private UnivariatePointValuePair[] optima;
    
    private OptimizationData[] optimData;
    
    private int maxEvalIndex = -1;
    
    private int searchIntervalIndex = -1;

    
    public MultiStartUnivariateOptimizer(final UnivariateOptimizer optimizer,
                                         final int starts,
                                         final RandomGenerator generator) {
        super(optimizer.getConvergenceChecker());

        if (starts < 1) {
            throw new NotStrictlyPositiveException(starts);
        }

        this.optimizer = optimizer;
        this.starts = starts;
        this.generator = generator;
    }

    
    @Override
    public int getEvaluations() {
        return totalEvaluations;
    }

    
    public UnivariatePointValuePair[] getOptima() {
        if (optima == null) {
            throw new MathIllegalStateException(LocalizedFormats.NO_OPTIMUM_COMPUTED_YET);
        }
        return optima.clone();
    }

    
    @Override
    public UnivariatePointValuePair optimize(OptimizationData... optData) {
        // Store arguments in order to pass them to the internal optimizer.
       optimData = optData;
        // Set up base class and perform computations.
        return super.optimize(optData);
    }

    
    @Override
    protected UnivariatePointValuePair doOptimize() {
        // Remove all instances of "MaxEval" and "SearchInterval" from the
        // array that will be passed to the internal optimizer.
        // The former is to enforce smaller numbers of allowed evaluations
        // (according to how many have been used up already), and the latter
        // to impose a different start value for each start.
        for (int i = 0; i < optimData.length; i++) {
            if (optimData[i] instanceof MaxEval) {
                optimData[i] = null;
                maxEvalIndex = i;
                continue;
            }
            if (optimData[i] instanceof SearchInterval) {
                optimData[i] = null;
                searchIntervalIndex = i;
                continue;
            }
        }
        if (maxEvalIndex == -1) {
            throw new MathIllegalStateException();
        }
        if (searchIntervalIndex == -1) {
            throw new MathIllegalStateException();
        }

        RuntimeException lastException = null;
        optima = new UnivariatePointValuePair[starts];
        totalEvaluations = 0;

        final int maxEval = getMaxEvaluations();
        final double min = getMin();
        final double max = getMax();
        final double startValue = getStartValue();

        // Multi-start loop.
        for (int i = 0; i < starts; i++) {
            // CHECKSTYLE: stop IllegalCatch
            try {
                // Decrease number of allowed evaluations.
                optimData[maxEvalIndex] = new MaxEval(maxEval - totalEvaluations);
                // New start value.
                final double s = (i == 0) ?
                    startValue :
                    min + generator.nextDouble() * (max - min);
                optimData[searchIntervalIndex] = new SearchInterval(min, max, s);
                // Optimize.
                optima[i] = optimizer.optimize(optimData);
            } catch (RuntimeException mue) {
                lastException = mue;
                optima[i] = null;
            }
            // CHECKSTYLE: resume IllegalCatch

            totalEvaluations += optimizer.getEvaluations();
        }

        sortPairs(getGoalType());

        if (optima[0] == null) {
            throw lastException; // Cannot be null if starts >= 1.
        }

        // Return the point with the best objective function value.
        return optima[0];
    }

    
    private void sortPairs(final GoalType goal) {
        Arrays.sort(optima, new Comparator<UnivariatePointValuePair>() {
                
                public int compare(final UnivariatePointValuePair o1,
                                   final UnivariatePointValuePair o2) {
                    if (o1 == null) {
                        return (o2 == null) ? 0 : 1;
                    } else if (o2 == null) {
                        return -1;
                    }
                    final double v1 = o1.getValue();
                    final double v2 = o2.getValue();
                    return (goal == GoalType.MINIMIZE) ?
                        Double.compare(v1, v2) : Double.compare(v2, v1);
                }
            });
    }
}
