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

import org.apache.lucene.util.hnsw.math.util.Incrementor;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.TooManyIterationsException;


public abstract class BaseOptimizer<PAIR> {
    
    protected final Incrementor evaluations;
    
    protected final Incrementor iterations;
    
    private final ConvergenceChecker<PAIR> checker;

    
    protected BaseOptimizer(ConvergenceChecker<PAIR> checker) {
        this(checker, 0, Integer.MAX_VALUE);
    }

    
    protected BaseOptimizer(ConvergenceChecker<PAIR> checker,
                            int maxEval,
                            int maxIter) {
        this.checker = checker;

        evaluations = new Incrementor(maxEval, new MaxEvalCallback());
        iterations = new Incrementor(maxIter, new MaxIterCallback());
    }

    
    public int getMaxEvaluations() {
        return evaluations.getMaximalCount();
    }

    
    public int getEvaluations() {
        return evaluations.getCount();
    }

    
    public int getMaxIterations() {
        return iterations.getMaximalCount();
    }

    
    public int getIterations() {
        return iterations.getCount();
    }

    
    public ConvergenceChecker<PAIR> getConvergenceChecker() {
        return checker;
    }

    
    public PAIR optimize(OptimizationData... optData)
        throws TooManyEvaluationsException,
               TooManyIterationsException {
        // Parse options.
        parseOptimizationData(optData);

        // Reset counters.
        evaluations.resetCount();
        iterations.resetCount();
        // Perform optimization.
        return doOptimize();
    }

    
    public PAIR optimize()
        throws TooManyEvaluationsException,
               TooManyIterationsException {
        // Reset counters.
        evaluations.resetCount();
        iterations.resetCount();
        // Perform optimization.
        return doOptimize();
    }

    
    protected abstract PAIR doOptimize();

    
    protected void incrementEvaluationCount()
        throws TooManyEvaluationsException {
        evaluations.incrementCount();
    }

    
    protected void incrementIterationCount()
        throws TooManyIterationsException {
        iterations.incrementCount();
    }

    
    protected void parseOptimizationData(OptimizationData... optData) {
        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof MaxEval) {
                evaluations.setMaximalCount(((MaxEval) data).getMaxEval());
                continue;
            }
            if (data instanceof MaxIter) {
                iterations.setMaximalCount(((MaxIter) data).getMaxIter());
                continue;
            }
        }
    }

    
    private static class MaxEvalCallback
        implements  Incrementor.MaxCountExceededCallback {
        
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
