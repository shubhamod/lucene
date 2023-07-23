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

package org.apache.lucene.util.hnsw.math.optimization.general;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.solvers.BrentSolver;
import org.apache.lucene.util.hnsw.math.analysis.solvers.UnivariateSolver;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.optimization.GoalType;
import org.apache.lucene.util.hnsw.math.optimization.PointValuePair;
import org.apache.lucene.util.hnsw.math.optimization.SimpleValueChecker;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.util.FastMath;


@Deprecated
public class NonLinearConjugateGradientOptimizer
    extends AbstractScalarDifferentiableOptimizer {
    
    private final ConjugateGradientFormula updateFormula;
    
    private final Preconditioner preconditioner;
    
    private final UnivariateSolver solver;
    
    private double initialStep;
    
    private double[] point;

    
    @Deprecated
    public NonLinearConjugateGradientOptimizer(final ConjugateGradientFormula updateFormula) {
        this(updateFormula,
             new SimpleValueChecker());
    }

    
    public NonLinearConjugateGradientOptimizer(final ConjugateGradientFormula updateFormula,
                                               ConvergenceChecker<PointValuePair> checker) {
        this(updateFormula,
             checker,
             new BrentSolver(),
             new IdentityPreconditioner());
    }


    
    public NonLinearConjugateGradientOptimizer(final ConjugateGradientFormula updateFormula,
                                               ConvergenceChecker<PointValuePair> checker,
                                               final UnivariateSolver lineSearchSolver) {
        this(updateFormula,
             checker,
             lineSearchSolver,
             new IdentityPreconditioner());
    }

    
    public NonLinearConjugateGradientOptimizer(final ConjugateGradientFormula updateFormula,
                                               ConvergenceChecker<PointValuePair> checker,
                                               final UnivariateSolver lineSearchSolver,
                                               final Preconditioner preconditioner) {
        super(checker);

        this.updateFormula = updateFormula;
        solver = lineSearchSolver;
        this.preconditioner = preconditioner;
        initialStep = 1.0;
    }

    
    public void setInitialStep(final double initialStep) {
        if (initialStep <= 0) {
            this.initialStep = 1.0;
        } else {
            this.initialStep = initialStep;
        }
    }

    
    @Override
    protected PointValuePair doOptimize() {
        final ConvergenceChecker<PointValuePair> checker = getConvergenceChecker();
        point = getStartPoint();
        final GoalType goal = getGoalType();
        final int n = point.length;
        double[] r = computeObjectiveGradient(point);
        if (goal == GoalType.MINIMIZE) {
            for (int i = 0; i < n; ++i) {
                r[i] = -r[i];
            }
        }

        // Initial search direction.
        double[] steepestDescent = preconditioner.precondition(point, r);
        double[] searchDirection = steepestDescent.clone();

        double delta = 0;
        for (int i = 0; i < n; ++i) {
            delta += r[i] * searchDirection[i];
        }

        PointValuePair current = null;
        int iter = 0;
        int maxEval = getMaxEvaluations();
        while (true) {
            ++iter;

            final double objective = computeObjectiveValue(point);
            PointValuePair previous = current;
            current = new PointValuePair(point, objective);
            if (previous != null && checker.converged(iter, previous, current)) {
                // We have found an optimum.
                return current;
            }

            // Find the optimal step in the search direction.
            final UnivariateFunction lsf = new LineSearchFunction(searchDirection);
            final double uB = findUpperBound(lsf, 0, initialStep);
            // XXX Last parameters is set to a value close to zero in order to
            // work around the divergence problem in the "testCircleFitting"
            // unit test (see MATH-439).
            final double step = solver.solve(maxEval, lsf, 0, uB, 1e-15);
            maxEval -= solver.getEvaluations(); // Subtract used up evaluations.

            // Validate new point.
            for (int i = 0; i < point.length; ++i) {
                point[i] += step * searchDirection[i];
            }

            r = computeObjectiveGradient(point);
            if (goal == GoalType.MINIMIZE) {
                for (int i = 0; i < n; ++i) {
                    r[i] = -r[i];
                }
            }

            // Compute beta.
            final double deltaOld = delta;
            final double[] newSteepestDescent = preconditioner.precondition(point, r);
            delta = 0;
            for (int i = 0; i < n; ++i) {
                delta += r[i] * newSteepestDescent[i];
            }

            final double beta;
            if (updateFormula == ConjugateGradientFormula.FLETCHER_REEVES) {
                beta = delta / deltaOld;
            } else {
                double deltaMid = 0;
                for (int i = 0; i < r.length; ++i) {
                    deltaMid += r[i] * steepestDescent[i];
                }
                beta = (delta - deltaMid) / deltaOld;
            }
            steepestDescent = newSteepestDescent;

            // Compute conjugate search direction.
            if (iter % n == 0 ||
                beta < 0) {
                // Break conjugation: reset search direction.
                searchDirection = steepestDescent.clone();
            } else {
                // Compute new conjugate search direction.
                for (int i = 0; i < n; ++i) {
                    searchDirection[i] = steepestDescent[i] + beta * searchDirection[i];
                }
            }
        }
    }

    
    private double findUpperBound(final UnivariateFunction f,
                                  final double a, final double h) {
        final double yA = f.value(a);
        double yB = yA;
        for (double step = h; step < Double.MAX_VALUE; step *= FastMath.max(2, yA / yB)) {
            final double b = a + step;
            yB = f.value(b);
            if (yA * yB <= 0) {
                return b;
            }
        }
        throw new MathIllegalStateException(LocalizedFormats.UNABLE_TO_BRACKET_OPTIMUM_IN_LINE_SEARCH);
    }

    
    public static class IdentityPreconditioner implements Preconditioner {

        
        public double[] precondition(double[] variables, double[] r) {
            return r.clone();
        }
    }

    
    private class LineSearchFunction implements UnivariateFunction {
        
        private final double[] searchDirection;

        
        LineSearchFunction(final double[] searchDirection) {
            this.searchDirection = searchDirection;
        }

        
        public double value(double x) {
            // current point in the search direction
            final double[] shiftedPoint = point.clone();
            for (int i = 0; i < shiftedPoint.length; ++i) {
                shiftedPoint[i] += x * searchDirection[i];
            }

            // gradient of the objective function
            final double[] gradient = computeObjectiveGradient(shiftedPoint);

            // dot product with the search direction
            double dotProduct = 0;
            for (int i = 0; i < gradient.length; ++i) {
                dotProduct += gradient[i] * searchDirection[i];
            }

            return dotProduct;
        }
    }
}
