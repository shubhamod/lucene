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
package org.apache.lucene.util.hnsw.math.optim;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.random.RandomVectorGenerator;


public abstract class BaseMultiStartMultivariateOptimizer<PAIR>
    extends BaseMultivariateOptimizer<PAIR> {
    
    private final BaseMultivariateOptimizer<PAIR> optimizer;
    
    private int totalEvaluations;
    
    private int starts;
    
    private RandomVectorGenerator generator;
    
    private OptimizationData[] optimData;
    
    private int maxEvalIndex = -1;
    
    private int initialGuessIndex = -1;

    
    public BaseMultiStartMultivariateOptimizer(final BaseMultivariateOptimizer<PAIR> optimizer,
                                               final int starts,
                                               final RandomVectorGenerator generator) {
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

    
    public abstract PAIR[] getOptima();

    
    @Override
    public PAIR optimize(OptimizationData... optData) {
        // Store arguments in order to pass them to the internal optimizer.
       optimData = optData;
        // Set up base class and perform computations.
        return super.optimize(optData);
    }

    
    @Override
    protected PAIR doOptimize() {
        // Remove all instances of "MaxEval" and "InitialGuess" from the
        // array that will be passed to the internal optimizer.
        // The former is to enforce smaller numbers of allowed evaluations
        // (according to how many have been used up already), and the latter
        // to impose a different start value for each start.
        for (int i = 0; i < optimData.length; i++) {
            if (optimData[i] instanceof MaxEval) {
                optimData[i] = null;
                maxEvalIndex = i;
            }
            if (optimData[i] instanceof InitialGuess) {
                optimData[i] = null;
                initialGuessIndex = i;
                continue;
            }
        }
        if (maxEvalIndex == -1) {
            throw new MathIllegalStateException();
        }
        if (initialGuessIndex == -1) {
            throw new MathIllegalStateException();
        }

        RuntimeException lastException = null;
        totalEvaluations = 0;
        clear();

        final int maxEval = getMaxEvaluations();
        final double[] min = getLowerBound();
        final double[] max = getUpperBound();
        final double[] startPoint = getStartPoint();

        // Multi-start loop.
        for (int i = 0; i < starts; i++) {
            // CHECKSTYLE: stop IllegalCatch
            try {
                // Decrease number of allowed evaluations.
                optimData[maxEvalIndex] = new MaxEval(maxEval - totalEvaluations);
                // New start value.
                double[] s = null;
                if (i == 0) {
                    s = startPoint;
                } else {
                    int attempts = 0;
                    while (s == null) {
                        if (attempts++ >= getMaxEvaluations()) {
                            throw new TooManyEvaluationsException(getMaxEvaluations());
                        }
                        s = generator.nextVector();
                        for (int k = 0; s != null && k < s.length; ++k) {
                            if ((min != null && s[k] < min[k]) || (max != null && s[k] > max[k])) {
                                // reject the vector
                                s = null;
                            }
                        }
                    }
                }
                optimData[initialGuessIndex] = new InitialGuess(s);
                // Optimize.
                final PAIR result = optimizer.optimize(optimData);
                store(result);
            } catch (RuntimeException mue) {
                lastException = mue;
            }
            // CHECKSTYLE: resume IllegalCatch

            totalEvaluations += optimizer.getEvaluations();
        }

        final PAIR[] optima = getOptima();
        if (optima.length == 0) {
            // All runs failed.
            throw lastException; // Cannot be null if starts >= 1.
        }

        // Return the best optimum.
        return optima[0];
    }

    
    protected abstract void store(PAIR optimum);
    
    protected abstract void clear();
}
