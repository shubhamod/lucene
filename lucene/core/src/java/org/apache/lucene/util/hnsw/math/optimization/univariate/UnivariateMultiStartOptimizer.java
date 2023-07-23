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

import java.util.Arrays;
import java.util.Comparator;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.optimization.GoalType;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;


@Deprecated
public class UnivariateMultiStartOptimizer<FUNC extends UnivariateFunction>
    implements BaseUnivariateOptimizer<FUNC> {
    
    private final BaseUnivariateOptimizer<FUNC> optimizer;
    
    private int maxEvaluations;
    
    private int totalEvaluations;
    
    private int starts;
    
    private RandomGenerator generator;
    
    private UnivariatePointValuePair[] optima;

    
    public UnivariateMultiStartOptimizer(final BaseUnivariateOptimizer<FUNC> optimizer,
                                             final int starts,
                                             final RandomGenerator generator) {
        if (optimizer == null ||
                generator == null) {
                throw new NullArgumentException();
        }
        if (starts < 1) {
            throw new NotStrictlyPositiveException(starts);
        }

        this.optimizer = optimizer;
        this.starts = starts;
        this.generator = generator;
    }

    
    public ConvergenceChecker<UnivariatePointValuePair> getConvergenceChecker() {
        return optimizer.getConvergenceChecker();
    }

    
    public int getMaxEvaluations() {
        return maxEvaluations;
    }

    
    public int getEvaluations() {
        return totalEvaluations;
    }

    
    public UnivariatePointValuePair[] getOptima() {
        if (optima == null) {
            throw new MathIllegalStateException(LocalizedFormats.NO_OPTIMUM_COMPUTED_YET);
        }
        return optima.clone();
    }

    
    public UnivariatePointValuePair optimize(int maxEval, final FUNC f,
                                                 final GoalType goal,
                                                 final double min, final double max) {
        return optimize(maxEval, f, goal, min, max, min + 0.5 * (max - min));
    }

    
    public UnivariatePointValuePair optimize(int maxEval, final FUNC f,
                                                 final GoalType goal,
                                                 final double min, final double max,
                                                 final double startValue) {
        RuntimeException lastException = null;
        optima = new UnivariatePointValuePair[starts];
        totalEvaluations = 0;

        // Multi-start loop.
        for (int i = 0; i < starts; ++i) {
            // CHECKSTYLE: stop IllegalCatch
            try {
                final double s = (i == 0) ? startValue : min + generator.nextDouble() * (max - min);
                optima[i] = optimizer.optimize(maxEval - totalEvaluations, f, goal, min, max, s);
            } catch (RuntimeException mue) {
                lastException = mue;
                optima[i] = null;
            }
            // CHECKSTYLE: resume IllegalCatch

            totalEvaluations += optimizer.getEvaluations();
        }

        sortPairs(goal);

        if (optima[0] == null) {
            throw lastException; // cannot be null if starts >=1
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
