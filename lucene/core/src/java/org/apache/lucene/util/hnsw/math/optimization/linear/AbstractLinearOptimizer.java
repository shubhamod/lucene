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

package org.apache.lucene.util.hnsw.math.optimization.linear;

import java.util.Collection;
import java.util.Collections;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.optimization.GoalType;
import org.apache.lucene.util.hnsw.math.optimization.PointValuePair;


@Deprecated
public abstract class AbstractLinearOptimizer implements LinearOptimizer {

    
    public static final int DEFAULT_MAX_ITERATIONS = 100;

    
    private LinearObjectiveFunction function;

    
    private Collection<LinearConstraint> linearConstraints;

    
    private GoalType goal;

    
    private boolean nonNegative;

    
    private int maxIterations;

    
    private int iterations;

    
    protected AbstractLinearOptimizer() {
        setMaxIterations(DEFAULT_MAX_ITERATIONS);
    }

    
    protected boolean restrictToNonNegative() {
        return nonNegative;
    }

    
    protected GoalType getGoalType() {
        return goal;
    }

    
    protected LinearObjectiveFunction getFunction() {
        return function;
    }

    
    protected Collection<LinearConstraint> getConstraints() {
        return Collections.unmodifiableCollection(linearConstraints);
    }

    
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    
    public int getMaxIterations() {
        return maxIterations;
    }

    
    public int getIterations() {
        return iterations;
    }

    
    protected void incrementIterationsCounter()
        throws MaxCountExceededException {
        if (++iterations > maxIterations) {
            throw new MaxCountExceededException(maxIterations);
        }
    }

    
    public PointValuePair optimize(final LinearObjectiveFunction f,
                                   final Collection<LinearConstraint> constraints,
                                   final GoalType goalType, final boolean restrictToNonNegative)
        throws MathIllegalStateException {

        // store linear problem characteristics
        this.function          = f;
        this.linearConstraints = constraints;
        this.goal              = goalType;
        this.nonNegative       = restrictToNonNegative;

        iterations  = 0;

        // solve the problem
        return doOptimize();

    }

    
    protected abstract PointValuePair doOptimize() throws MathIllegalStateException;

}
