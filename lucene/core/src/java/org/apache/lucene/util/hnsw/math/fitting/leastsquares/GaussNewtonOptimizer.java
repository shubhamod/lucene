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

import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem.Evaluation;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.CholeskyDecomposition;
import org.apache.lucene.util.hnsw.math.linear.LUDecomposition;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.NonPositiveDefiniteMatrixException;
import org.apache.lucene.util.hnsw.math.linear.QRDecomposition;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.linear.SingularMatrixException;
import org.apache.lucene.util.hnsw.math.linear.SingularValueDecomposition;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.util.Incrementor;
import org.apache.lucene.util.hnsw.math.util.Pair;


public class GaussNewtonOptimizer implements LeastSquaresOptimizer {

    
    //TODO move to linear package and expand options?
    public enum Decomposition {
        
        LU {
            @Override
            protected RealVector solve(final RealMatrix jacobian,
                                       final RealVector residuals) {
                try {
                    final Pair<RealMatrix, RealVector> normalEquation =
                            computeNormalMatrix(jacobian, residuals);
                    final RealMatrix normal = normalEquation.getFirst();
                    final RealVector jTr = normalEquation.getSecond();
                    return new LUDecomposition(normal, SINGULARITY_THRESHOLD)
                            .getSolver()
                            .solve(jTr);
                } catch (SingularMatrixException e) {
                    throw new ConvergenceException(LocalizedFormats.UNABLE_TO_SOLVE_SINGULAR_PROBLEM, e);
                }
            }
        },
        
        QR {
            @Override
            protected RealVector solve(final RealMatrix jacobian,
                                       final RealVector residuals) {
                try {
                    return new QRDecomposition(jacobian, SINGULARITY_THRESHOLD)
                            .getSolver()
                            .solve(residuals);
                } catch (SingularMatrixException e) {
                    throw new ConvergenceException(LocalizedFormats.UNABLE_TO_SOLVE_SINGULAR_PROBLEM, e);
                }
            }
        },
        
        CHOLESKY {
            @Override
            protected RealVector solve(final RealMatrix jacobian,
                                       final RealVector residuals) {
                try {
                    final Pair<RealMatrix, RealVector> normalEquation =
                            computeNormalMatrix(jacobian, residuals);
                    final RealMatrix normal = normalEquation.getFirst();
                    final RealVector jTr = normalEquation.getSecond();
                    return new CholeskyDecomposition(
                            normal, SINGULARITY_THRESHOLD, SINGULARITY_THRESHOLD)
                            .getSolver()
                            .solve(jTr);
                } catch (NonPositiveDefiniteMatrixException e) {
                    throw new ConvergenceException(LocalizedFormats.UNABLE_TO_SOLVE_SINGULAR_PROBLEM, e);
                }
            }
        },
        
        SVD {
            @Override
            protected RealVector solve(final RealMatrix jacobian,
                                       final RealVector residuals) {
                return new SingularValueDecomposition(jacobian)
                        .getSolver()
                        .solve(residuals);
            }
        };

        
        protected abstract RealVector solve(RealMatrix jacobian,
                                            RealVector residuals);
    }

    
    private static final double SINGULARITY_THRESHOLD = 1e-11;

    
    private final Decomposition decomposition;

    
    public GaussNewtonOptimizer() {
        this(Decomposition.QR);
    }

    
    public GaussNewtonOptimizer(final Decomposition decomposition) {
        this.decomposition = decomposition;
    }

    
    public Decomposition getDecomposition() {
        return this.decomposition;
    }

    
    public GaussNewtonOptimizer withDecomposition(final Decomposition newDecomposition) {
        return new GaussNewtonOptimizer(newDecomposition);
    }

    
    public Optimum optimize(final LeastSquaresProblem lsp) {
        //create local evaluation and iteration counts
        final Incrementor evaluationCounter = lsp.getEvaluationCounter();
        final Incrementor iterationCounter = lsp.getIterationCounter();
        final ConvergenceChecker<Evaluation> checker
                = lsp.getConvergenceChecker();

        // Computation will be useless without a checker (see "for-loop").
        if (checker == null) {
            throw new NullArgumentException();
        }

        RealVector currentPoint = lsp.getStart();

        // iterate until convergence is reached
        Evaluation current = null;
        while (true) {
            iterationCounter.incrementCount();

            // evaluate the objective function and its jacobian
            Evaluation previous = current;
            // Value of the objective function at "currentPoint".
            evaluationCounter.incrementCount();
            current = lsp.evaluate(currentPoint);
            final RealVector currentResiduals = current.getResiduals();
            final RealMatrix weightedJacobian = current.getJacobian();
            currentPoint = current.getPoint();

            // Check convergence.
            if (previous != null &&
                checker.converged(iterationCounter.getCount(), previous, current)) {
                return new OptimumImpl(current,
                                       evaluationCounter.getCount(),
                                       iterationCounter.getCount());
            }

            // solve the linearized least squares problem
            final RealVector dX = this.decomposition.solve(weightedJacobian, currentResiduals);
            // update the estimated parameters
            currentPoint = currentPoint.add(dX);
        }
    }

    
    @Override
    public String toString() {
        return "GaussNewtonOptimizer{" +
                "decomposition=" + decomposition +
                '}';
    }

    
    private static Pair<RealMatrix, RealVector> computeNormalMatrix(final RealMatrix jacobian,
                                                                    final RealVector residuals) {
        //since the normal matrix is symmetric, we only need to compute half of it.
        final int nR = jacobian.getRowDimension();
        final int nC = jacobian.getColumnDimension();
        //allocate space for return values
        final RealMatrix normal = MatrixUtils.createRealMatrix(nC, nC);
        final RealVector jTr = new ArrayRealVector(nC);
        //for each measurement
        for (int i = 0; i < nR; ++i) {
            //compute JTr for measurement i
            for (int j = 0; j < nC; j++) {
                jTr.setEntry(j, jTr.getEntry(j) +
                        residuals.getEntry(i) * jacobian.getEntry(i, j));
            }

            // add the the contribution to the normal matrix for measurement i
            for (int k = 0; k < nC; ++k) {
                //only compute the upper triangular part
                for (int l = k; l < nC; ++l) {
                    normal.setEntry(k, l, normal.getEntry(k, l) +
                            jacobian.getEntry(i, k) * jacobian.getEntry(i, l));
                }
            }
        }
        //copy the upper triangular part to the lower triangular part.
        for (int i = 0; i < nC; i++) {
            for (int j = 0; j < i; j++) {
                normal.setEntry(i, j, normal.getEntry(j, i));
            }
        }
        return new Pair<RealMatrix, RealVector>(normal, jTr);
    }

}
