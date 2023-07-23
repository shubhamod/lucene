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
package org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.noderiv;

import java.util.Comparator;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathUnsupportedOperationException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.GoalType;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optim.PointValuePair;
import org.apache.lucene.util.hnsw.math.optim.SimpleValueChecker;
import org.apache.lucene.util.hnsw.math.optim.OptimizationData;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.MultivariateOptimizer;


public class SimplexOptimizer extends MultivariateOptimizer {
    
    private AbstractSimplex simplex;

    
    public SimplexOptimizer(ConvergenceChecker<PointValuePair> checker) {
        super(checker);
    }

    
    public SimplexOptimizer(double rel, double abs) {
        this(new SimpleValueChecker(rel, abs));
    }

    
    @Override
    public PointValuePair optimize(OptimizationData... optData) {
        // Set up base class and perform computation.
        return super.optimize(optData);
    }

    
    @Override
    protected PointValuePair doOptimize() {
        checkParameters();

        // Indirect call to "computeObjectiveValue" in order to update the
        // evaluations counter.
        final MultivariateFunction evalFunc
            = new MultivariateFunction() {
                
                public double value(double[] point) {
                    return computeObjectiveValue(point);
                }
            };

        final boolean isMinim = getGoalType() == GoalType.MINIMIZE;
        final Comparator<PointValuePair> comparator
            = new Comparator<PointValuePair>() {
            
            public int compare(final PointValuePair o1,
                               final PointValuePair o2) {
                final double v1 = o1.getValue();
                final double v2 = o2.getValue();
                return isMinim ? Double.compare(v1, v2) : Double.compare(v2, v1);
            }
        };

        // Initialize search.
        simplex.build(getStartPoint());
        simplex.evaluate(evalFunc, comparator);

        PointValuePair[] previous = null;
        int iteration = 0;
        final ConvergenceChecker<PointValuePair> checker = getConvergenceChecker();
        while (true) {
            if (getIterations() > 0) {
                boolean converged = true;
                for (int i = 0; i < simplex.getSize(); i++) {
                    PointValuePair prev = previous[i];
                    converged = converged &&
                        checker.converged(iteration, prev, simplex.getPoint(i));
                }
                if (converged) {
                    // We have found an optimum.
                    return simplex.getPoint(0);
                }
            }

            // We still need to search.
            previous = simplex.getPoints();
            simplex.iterate(evalFunc, comparator);

            incrementIterationCount();
        }
    }

    
    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        // Allow base class to register its own data.
        super.parseOptimizationData(optData);

        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof AbstractSimplex) {
                simplex = (AbstractSimplex) data;
                // If more data must be parsed, this statement _must_ be
                // changed to "continue".
                break;
            }
        }
    }

    
    private void checkParameters() {
        if (simplex == null) {
            throw new NullArgumentException();
        }
        if (getLowerBound() != null ||
            getUpperBound() != null) {
            throw new MathUnsupportedOperationException(LocalizedFormats.CONSTRAINT);
        }
    }
}
