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
package org.apache.lucene.util.hnsw.math.fitting.leastsquares;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateMatrixFunction;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem.Evaluation;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.DiagonalMatrix;
import org.apache.lucene.util.hnsw.math.linear.EigenDecomposition;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.optim.AbstractOptimizationProblem;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optim.PointVectorValuePair;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Incrementor;
import org.apache.lucene.util.hnsw.math.util.Pair;


public class LeastSquaresFactory {

    
    private LeastSquaresFactory() {}

    
    public static LeastSquaresProblem create(final MultivariateJacobianFunction model,
                                             final RealVector observed,
                                             final RealVector start,
                                             final RealMatrix weight,
                                             final ConvergenceChecker<Evaluation> checker,
                                             final int maxEvaluations,
                                             final int maxIterations,
                                             final boolean lazyEvaluation,
                                             final ParameterValidator paramValidator) {
        final LeastSquaresProblem p = new LocalLeastSquaresProblem(model,
                                                                   observed,
                                                                   start,
                                                                   checker,
                                                                   maxEvaluations,
                                                                   maxIterations,
                                                                   lazyEvaluation,
                                                                   paramValidator);
        if (weight != null) {
            return weightMatrix(p, weight);
        } else {
            return p;
        }
    }

    
    public static LeastSquaresProblem create(final MultivariateJacobianFunction model,
                                             final RealVector observed,
                                             final RealVector start,
                                             final ConvergenceChecker<Evaluation> checker,
                                             final int maxEvaluations,
                                             final int maxIterations) {
        return create(model,
                      observed,
                      start,
                      null,
                      checker,
                      maxEvaluations,
                      maxIterations,
                      false,
                      null);
    }

    
    public static LeastSquaresProblem create(final MultivariateJacobianFunction model,
                                             final RealVector observed,
                                             final RealVector start,
                                             final RealMatrix weight,
                                             final ConvergenceChecker<Evaluation> checker,
                                             final int maxEvaluations,
                                             final int maxIterations) {
        return weightMatrix(create(model,
                                   observed,
                                   start,
                                   checker,
                                   maxEvaluations,
                                   maxIterations),
                            weight);
    }

    
    public static LeastSquaresProblem create(final MultivariateVectorFunction model,
                                             final MultivariateMatrixFunction jacobian,
                                             final double[] observed,
                                             final double[] start,
                                             final RealMatrix weight,
                                             final ConvergenceChecker<Evaluation> checker,
                                             final int maxEvaluations,
                                             final int maxIterations) {
        return create(model(model, jacobian),
                      new ArrayRealVector(observed, false),
                      new ArrayRealVector(start, false),
                      weight,
                      checker,
                      maxEvaluations,
                      maxIterations);
    }

    
    public static LeastSquaresProblem weightMatrix(final LeastSquaresProblem problem,
                                                   final RealMatrix weights) {
        final RealMatrix weightSquareRoot = squareRoot(weights);
        return new LeastSquaresAdapter(problem) {
            
            @Override
            public Evaluation evaluate(final RealVector point) {
                return new DenseWeightedEvaluation(super.evaluate(point), weightSquareRoot);
            }
        };
    }

    
    public static LeastSquaresProblem weightDiagonal(final LeastSquaresProblem problem,
                                                     final RealVector weights) {
        // TODO more efficient implementation
        return weightMatrix(problem, new DiagonalMatrix(weights.toArray()));
    }

    
    public static LeastSquaresProblem countEvaluations(final LeastSquaresProblem problem,
                                                       final Incrementor counter) {
        return new LeastSquaresAdapter(problem) {

            
            @Override
            public Evaluation evaluate(final RealVector point) {
                counter.incrementCount();
                return super.evaluate(point);
            }

            // Delegate the rest.
        };
    }

    
    public static ConvergenceChecker<Evaluation> evaluationChecker(final ConvergenceChecker<PointVectorValuePair> checker) {
        return new ConvergenceChecker<Evaluation>() {
            
            public boolean converged(final int iteration,
                                     final Evaluation previous,
                                     final Evaluation current) {
                return checker.converged(
                        iteration,
                        new PointVectorValuePair(
                                previous.getPoint().toArray(),
                                previous.getResiduals().toArray(),
                                false),
                        new PointVectorValuePair(
                                current.getPoint().toArray(),
                                current.getResiduals().toArray(),
                                false)
                );
            }
        };
    }

    
    private static RealMatrix squareRoot(final RealMatrix m) {
        if (m instanceof DiagonalMatrix) {
            final int dim = m.getRowDimension();
            final RealMatrix sqrtM = new DiagonalMatrix(dim);
            for (int i = 0; i < dim; i++) {
                sqrtM.setEntry(i, i, FastMath.sqrt(m.getEntry(i, i)));
            }
            return sqrtM;
        } else {
            final EigenDecomposition dec = new EigenDecomposition(m);
            return dec.getSquareRoot();
        }
    }

    
    public static MultivariateJacobianFunction model(final MultivariateVectorFunction value,
                                                     final MultivariateMatrixFunction jacobian) {
        return new LocalValueAndJacobianFunction(value, jacobian);
    }

    
    private static class LocalValueAndJacobianFunction
        implements ValueAndJacobianFunction {
        
        private final MultivariateVectorFunction value;
        
        private final MultivariateMatrixFunction jacobian;

        
        LocalValueAndJacobianFunction(final MultivariateVectorFunction value,
                                      final MultivariateMatrixFunction jacobian) {
            this.value = value;
            this.jacobian = jacobian;
        }

        
        public Pair<RealVector, RealMatrix> value(final RealVector point) {
            //TODO get array from RealVector without copying?
            final double[] p = point.toArray();

            // Evaluate.
            return new Pair<RealVector, RealMatrix>(computeValue(p),
                                                    computeJacobian(p));
        }

        
        public RealVector computeValue(final double[] params) {
            return new ArrayRealVector(value.value(params), false);
        }

        
        public RealMatrix computeJacobian(final double[] params) {
            return new Array2DRowRealMatrix(jacobian.value(params), false);
        }
    }


    
    private static class LocalLeastSquaresProblem
            extends AbstractOptimizationProblem<Evaluation>
            implements LeastSquaresProblem {

        
        private final RealVector target;
        
        private final MultivariateJacobianFunction model;
        
        private final RealVector start;
        
        private final boolean lazyEvaluation;
        
        private final ParameterValidator paramValidator;

        
        LocalLeastSquaresProblem(final MultivariateJacobianFunction model,
                                 final RealVector target,
                                 final RealVector start,
                                 final ConvergenceChecker<Evaluation> checker,
                                 final int maxEvaluations,
                                 final int maxIterations,
                                 final boolean lazyEvaluation,
                                 final ParameterValidator paramValidator) {
            super(maxEvaluations, maxIterations, checker);
            this.target = target;
            this.model = model;
            this.start = start;
            this.lazyEvaluation = lazyEvaluation;
            this.paramValidator = paramValidator;

            if (lazyEvaluation &&
                !(model instanceof ValueAndJacobianFunction)) {
                // Lazy evaluation requires that value and Jacobian
                // can be computed separately.
                throw new MathIllegalStateException(LocalizedFormats.INVALID_IMPLEMENTATION,
                                                    model.getClass().getName());
            }
        }

        
        public int getObservationSize() {
            return target.getDimension();
        }

        
        public int getParameterSize() {
            return start.getDimension();
        }

        
        public RealVector getStart() {
            return start == null ? null : start.copy();
        }

        
        public Evaluation evaluate(final RealVector point) {
            // Copy so optimizer can change point without changing our instance.
            final RealVector p = paramValidator == null ?
                point.copy() :
                paramValidator.validate(point.copy());

            if (lazyEvaluation) {
                return new LazyUnweightedEvaluation((ValueAndJacobianFunction) model,
                                                    target,
                                                    p);
            } else {
                // Evaluate value and jacobian in one function call.
                final Pair<RealVector, RealMatrix> value = model.value(p);
                return new UnweightedEvaluation(value.getFirst(),
                                                value.getSecond(),
                                                target,
                                                p);
            }
        }

        
        private static class UnweightedEvaluation extends AbstractEvaluation {
            
            private final RealVector point;
            
            private final RealMatrix jacobian;
            
            private final RealVector residuals;

            
            private UnweightedEvaluation(final RealVector values,
                                         final RealMatrix jacobian,
                                         final RealVector target,
                                         final RealVector point) {
                super(target.getDimension());
                this.jacobian = jacobian;
                this.point = point;
                this.residuals = target.subtract(values);
            }

            
            public RealMatrix getJacobian() {
                return jacobian;
            }

            
            public RealVector getPoint() {
                return point;
            }

            
            public RealVector getResiduals() {
                return residuals;
            }
        }

        
        private static class LazyUnweightedEvaluation extends AbstractEvaluation {
            
            private final RealVector point;
            
            private final ValueAndJacobianFunction model;
            
            private final RealVector target;

            
            private LazyUnweightedEvaluation(final ValueAndJacobianFunction model,
                                             final RealVector target,
                                             final RealVector point) {
                super(target.getDimension());
                // Safe to cast as long as we control usage of this class.
                this.model = model;
                this.point = point;
                this.target = target;
            }

            
            public RealMatrix getJacobian() {
                return model.computeJacobian(point.toArray());
            }

            
            public RealVector getPoint() {
                return point;
            }

            
            public RealVector getResiduals() {
                return target.subtract(model.computeValue(point.toArray()));
            }
        }
    }
}

