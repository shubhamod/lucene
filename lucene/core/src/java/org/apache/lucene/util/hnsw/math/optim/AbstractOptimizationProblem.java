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

import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.TooManyIterationsException;
import org.apache.lucene.util.hnsw.math.util.Incrementor;


public abstract class AbstractOptimizationProblem<PAIR>
        implements OptimizationProblem<PAIR> {

    
    private static final MaxEvalCallback MAX_EVAL_CALLBACK = new MaxEvalCallback();
    
    private static final MaxIterCallback MAX_ITER_CALLBACK = new MaxIterCallback();

    
    private final int maxEvaluations;
    
    private final int maxIterations;
    
    private final ConvergenceChecker<PAIR> checker;

    
    protected AbstractOptimizationProblem(final int maxEvaluations,
                                          final int maxIterations,
                                          final ConvergenceChecker<PAIR> checker) {
        this.maxEvaluations = maxEvaluations;
        this.maxIterations = maxIterations;
        this.checker = checker;
    }

    
    public Incrementor getEvaluationCounter() {
        return new Incrementor(this.maxEvaluations, MAX_EVAL_CALLBACK);
    }

    
    public Incrementor getIterationCounter() {
        return new Incrementor(this.maxIterations, MAX_ITER_CALLBACK);
    }

    
    public ConvergenceChecker<PAIR> getConvergenceChecker() {
        return checker;
    }

    
    private static class MaxEvalCallback
            implements Incrementor.MaxCountExceededCallback {
        
        public void trigger(int max) {
            throw new TooManyEvaluationsException(max);
        }
    }

    
    private static class MaxIterCallback
            implements Incrementor.MaxCountExceededCallback {
        
        public void trigger(int max) {
            throw new TooManyIterationsException(max);
        }
    }

}
