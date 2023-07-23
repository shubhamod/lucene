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
package org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.jacobian;

import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.MathUnsupportedOperationException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.BlockRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.DecompositionSolver;
import org.apache.lucene.util.hnsw.math.linear.LUDecomposition;
import org.apache.lucene.util.hnsw.math.linear.QRDecomposition;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.SingularMatrixException;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optim.PointVectorValuePair;


@Deprecated
public class GaussNewtonOptimizer extends AbstractLeastSquaresOptimizer {
    
    private final boolean useLU;

    
    public GaussNewtonOptimizer(ConvergenceChecker<PointVectorValuePair> checker) {
        this(true, checker);
    }

    
    public GaussNewtonOptimizer(final boolean useLU,
                                ConvergenceChecker<PointVectorValuePair> checker) {
        super(checker);
        this.useLU = useLU;
    }

    
    @Override
    public PointVectorValuePair doOptimize() {
        checkParameters();

        final ConvergenceChecker<PointVectorValuePair> checker
            = getConvergenceChecker();

        // Computation will be useless without a checker (see "for-loop").
        if (checker == null) {
            throw new NullArgumentException();
        }

        final double[] targetValues = getTarget();
        final int nR = targetValues.length; // Number of observed data.

        final RealMatrix weightMatrix = getWeight();
        // Diagonal of the weight matrix.
        final double[] residualsWeights = new double[nR];
        for (int i = 0; i < nR; i++) {
            residualsWeights[i] = weightMatrix.getEntry(i, i);
        }

        final double[] currentPoint = getStartPoint();
        final int nC = currentPoint.length;

        // iterate until convergence is reached
        PointVectorValuePair current = null;
        for (boolean converged = false; !converged;) {
            incrementIterationCount();

            // evaluate the objective function and its jacobian
            PointVectorValuePair previous = current;
            // Value of the objective function at "currentPoint".
            final double[] currentObjective = computeObjectiveValue(currentPoint);
            final double[] currentResiduals = computeResiduals(currentObjective);
            final RealMatrix weightedJacobian = computeWeightedJacobian(currentPoint);
            current = new PointVectorValuePair(currentPoint, currentObjective);

            // build the linear problem
            final double[]   b = new double[nC];
            final double[][] a = new double[nC][nC];
            for (int i = 0; i < nR; ++i) {

                final double[] grad   = weightedJacobian.getRow(i);
                final double weight   = residualsWeights[i];
                final double residual = currentResiduals[i];

                // compute the normal equation
                final double wr = weight * residual;
                for (int j = 0; j < nC; ++j) {
                    b[j] += wr * grad[j];
                }

                // build the contribution matrix for measurement i
                for (int k = 0; k < nC; ++k) {
                    double[] ak = a[k];
                    double wgk = weight * grad[k];
                    for (int l = 0; l < nC; ++l) {
                        ak[l] += wgk * grad[l];
                    }
                }
            }

            // Check convergence.
            if (previous != null) {
                converged = checker.converged(getIterations(), previous, current);
                if (converged) {
                    setCost(computeCost(currentResiduals));
                    return current;
                }
            }

            try {
                // solve the linearized least squares problem
                RealMatrix mA = new BlockRealMatrix(a);
                DecompositionSolver solver = useLU ?
                        new LUDecomposition(mA).getSolver() :
                        new QRDecomposition(mA).getSolver();
                final double[] dX = solver.solve(new ArrayRealVector(b, false)).toArray();
                // update the estimated parameters
                for (int i = 0; i < nC; ++i) {
                    currentPoint[i] += dX[i];
                }
            } catch (SingularMatrixException e) {
                throw new ConvergenceException(LocalizedFormats.UNABLE_TO_SOLVE_SINGULAR_PROBLEM);
            }
        }
        // Must never happen.
        throw new MathInternalError();
    }

    
    private void checkParameters() {
        if (getLowerBound() != null ||
            getUpperBound() != null) {
            throw new MathUnsupportedOperationException(LocalizedFormats.CONSTRAINT);
        }
    }
}
