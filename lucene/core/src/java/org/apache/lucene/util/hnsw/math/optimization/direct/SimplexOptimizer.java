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

import java.util.Comparator;

import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.optimization.GoalType;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optimization.PointValuePair;
import org.apache.lucene.util.hnsw.math.optimization.SimpleValueChecker;
import org.apache.lucene.util.hnsw.math.optimization.MultivariateOptimizer;
import org.apache.lucene.util.hnsw.math.optimization.OptimizationData;


@SuppressWarnings("boxing") // deprecated anyway
@Deprecated
public class SimplexOptimizer
    extends BaseAbstractMultivariateOptimizer<MultivariateFunction>
    implements MultivariateOptimizer {
    
    private AbstractSimplex simplex;

    
    @Deprecated
    public SimplexOptimizer() {
        this(new SimpleValueChecker());
    }

    
    public SimplexOptimizer(ConvergenceChecker<PointValuePair> checker) {
        super(checker);
    }

    
    public SimplexOptimizer(double rel, double abs) {
        this(new SimpleValueChecker(rel, abs));
    }

    
    @Deprecated
    public void setSimplex(AbstractSimplex simplex) {
        parseOptimizationData(simplex);
    }

    
    @Override
    protected PointValuePair optimizeInternal(int maxEval, MultivariateFunction f,
                                              GoalType goalType,
                                              OptimizationData... optData) {
        // Scan "optData" for the input specific to this optimizer.
        parseOptimizationData(optData);

        // The parent's method will retrieve the common parameters from
        // "optData" and call "doOptimize".
        return super.optimizeInternal(maxEval, f, goalType, optData);
    }

    
    private void parseOptimizationData(OptimizationData... optData) {
        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof AbstractSimplex) {
                simplex = (AbstractSimplex) data;
                continue;
            }
        }
    }

    
    @Override
    protected PointValuePair doOptimize() {
        if (simplex == null) {
            throw new NullArgumentException();
        }

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
            if (iteration > 0) {
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
            ++iteration;
        }
    }
}
