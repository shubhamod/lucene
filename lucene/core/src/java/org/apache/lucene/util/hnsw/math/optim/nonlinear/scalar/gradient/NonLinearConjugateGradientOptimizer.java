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

package org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.gradient;

import org.apache.lucene.util.hnsw.math.analysis.solvers.UnivariateSolver;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.MathUnsupportedOperationException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.optim.OptimizationData;
import org.apache.lucene.util.hnsw.math.optim.PointValuePair;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.GoalType;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.GradientMultivariateOptimizer;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.LineSearch;



public class NonLinearConjugateGradientOptimizer
    extends GradientMultivariateOptimizer {
    
    private final Formula updateFormula;
    
    private final Preconditioner preconditioner;
    
    private final LineSearch line;

    
    public enum Formula {
        
        FLETCHER_REEVES,
        
        POLAK_RIBIERE
    }

    
    @Deprecated
    public static class BracketingStep implements OptimizationData {
        
        private final double initialStep;

        
        public BracketingStep(double step) {
            initialStep = step;
        }

        
        public double getBracketingStep() {
            return initialStep;
        }
    }

    
    public NonLinearConjugateGradientOptimizer(final Formula updateFormula,
                                               ConvergenceChecker<PointValuePair> checker) {
        this(updateFormula,
             checker,
             1e-8,
             1e-8,
             1e-8,
             new IdentityPreconditioner());
    }

    
    @Deprecated
    public NonLinearConjugateGradientOptimizer(final Formula updateFormula,
                                               ConvergenceChecker<PointValuePair> checker,
                                               final UnivariateSolver lineSearchSolver) {
        this(updateFormula,
             checker,
             lineSearchSolver,
             new IdentityPreconditioner());
    }

    
    public NonLinearConjugateGradientOptimizer(final Formula updateFormula,
                                               ConvergenceChecker<PointValuePair> checker,
                                               double relativeTolerance,
                                               double absoluteTolerance,
                                               double initialBracketingRange) {
        this(updateFormula,
             checker,
             relativeTolerance,
             absoluteTolerance,
             initialBracketingRange,
             new IdentityPreconditioner());
    }

    
    @Deprecated
    public NonLinearConjugateGradientOptimizer(final Formula updateFormula,
                                               ConvergenceChecker<PointValuePair> checker,
                                               final UnivariateSolver lineSearchSolver,
                                               final Preconditioner preconditioner) {
        this(updateFormula,
             checker,
             lineSearchSolver.getRelativeAccuracy(),
             lineSearchSolver.getAbsoluteAccuracy(),
             lineSearchSolver.getAbsoluteAccuracy(),
             preconditioner);
    }

    
    public NonLinearConjugateGradientOptimizer(final Formula updateFormula,
                                               ConvergenceChecker<PointValuePair> checker,
                                               double relativeTolerance,
                                               double absoluteTolerance,
                                               double initialBracketingRange,
                                               final Preconditioner preconditioner) {
        super(checker);

        this.updateFormula = updateFormula;
        this.preconditioner = preconditioner;
        line = new LineSearch(this,
                              relativeTolerance,
                              absoluteTolerance,
                              initialBracketingRange);
    }

    
    @Override
    public PointValuePair optimize(OptimizationData... optData)
        throws TooManyEvaluationsException {
        // Set up base class and perform computation.
        return super.optimize(optData);
    }

    
    @Override
    protected PointValuePair doOptimize() {
        final ConvergenceChecker<PointValuePair> checker = getConvergenceChecker();
        final double[] point = getStartPoint();
        final GoalType goal = getGoalType();
        final int n = point.length;
        double[] r = computeObjectiveGradient(point);
        if (goal == GoalType.MINIMIZE) {
            for (int i = 0; i < n; i++) {
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
        while (true) {
            incrementIterationCount();

            final double objective = computeObjectiveValue(point);
            PointValuePair previous = current;
            current = new PointValuePair(point, objective);
            if (previous != null && checker.converged(getIterations(), previous, current)) {
                // We have found an optimum.
                return current;
            }

            final double step = line.search(point, searchDirection).getPoint();

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
            switch (updateFormula) {
            case FLETCHER_REEVES:
                beta = delta / deltaOld;
                break;
            case POLAK_RIBIERE:
                double deltaMid = 0;
                for (int i = 0; i < r.length; ++i) {
                    deltaMid += r[i] * steepestDescent[i];
                }
                beta = (delta - deltaMid) / deltaOld;
                break;
            default:
                // Should never happen.
                throw new MathInternalError();
            }
            steepestDescent = newSteepestDescent;

            // Compute conjugate search direction.
            if (getIterations() % n == 0 ||
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

    
    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        // Allow base class to register its own data.
        super.parseOptimizationData(optData);

        checkParameters();
    }

    
    public static class IdentityPreconditioner implements Preconditioner {
        
        public double[] precondition(double[] variables, double[] r) {
            return r.clone();
        }
    }

    // Class is not used anymore (cf. MATH-1092). However, it might
    // be interesting to create a class similar to "LineSearch", but
    // that will take advantage that the model's gradient is available.
//     
//     private class LineSearchFunction implements UnivariateFunction {
//         
//         private final double[] currentPoint;
//         
//         private final double[] searchDirection;

//         
//         public LineSearchFunction(double[] point,
//                                   double[] direction) {
//             currentPoint = point.clone();
//             searchDirection = direction.clone();
//         }

//         
//         public double value(double x) {
//             // current point in the search direction
//             final double[] shiftedPoint = currentPoint.clone();
//             for (int i = 0; i < shiftedPoint.length; ++i) {
//                 shiftedPoint[i] += x * searchDirection[i];
//             }

//             // gradient of the objective function
//             final double[] gradient = computeObjectiveGradient(shiftedPoint);

//             // dot product with the search direction
//             double dotProduct = 0;
//             for (int i = 0; i < gradient.length; ++i) {
//                 dotProduct += gradient[i] * searchDirection[i];
//             }

//             return dotProduct;
//         }
//     }

    
    private void checkParameters() {
        if (getLowerBound() != null ||
            getUpperBound() != null) {
            throw new MathUnsupportedOperationException(LocalizedFormats.CONSTRAINT);
        }
    }
}
