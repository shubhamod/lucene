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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.DiagonalMatrix;
import org.apache.lucene.util.hnsw.math.linear.DecompositionSolver;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.QRDecomposition;
import org.apache.lucene.util.hnsw.math.linear.EigenDecomposition;
import org.apache.lucene.util.hnsw.math.optim.OptimizationData;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optim.PointVectorValuePair;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.Weight;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.JacobianMultivariateVectorOptimizer;
import org.apache.lucene.util.hnsw.math.util.FastMath;


@Deprecated
public abstract class AbstractLeastSquaresOptimizer
    extends JacobianMultivariateVectorOptimizer {
    
    private RealMatrix weightMatrixSqrt;
    
    private double cost;

    
    protected AbstractLeastSquaresOptimizer(ConvergenceChecker<PointVectorValuePair> checker) {
        super(checker);
    }

    
    protected RealMatrix computeWeightedJacobian(double[] params) {
        return weightMatrixSqrt.multiply(MatrixUtils.createRealMatrix(computeJacobian(params)));
    }

    
    protected double computeCost(double[] residuals) {
        final ArrayRealVector r = new ArrayRealVector(residuals);
        return FastMath.sqrt(r.dotProduct(getWeight().operate(r)));
    }

    
    public double getRMS() {
        return FastMath.sqrt(getChiSquare() / getTargetSize());
    }

    
    public double getChiSquare() {
        return cost * cost;
    }

    
    public RealMatrix getWeightSquareRoot() {
        return weightMatrixSqrt.copy();
    }

    
    protected void setCost(double cost) {
        this.cost = cost;
    }

    
    public double[][] computeCovariances(double[] params,
                                         double threshold) {
        // Set up the Jacobian.
        final RealMatrix j = computeWeightedJacobian(params);

        // Compute transpose(J)J.
        final RealMatrix jTj = j.transpose().multiply(j);

        // Compute the covariances matrix.
        final DecompositionSolver solver
            = new QRDecomposition(jTj, threshold).getSolver();
        return solver.getInverse().getData();
    }

    
    public double[] computeSigma(double[] params,
                                 double covarianceSingularityThreshold) {
        final int nC = params.length;
        final double[] sig = new double[nC];
        final double[][] cov = computeCovariances(params, covarianceSingularityThreshold);
        for (int i = 0; i < nC; ++i) {
            sig[i] = FastMath.sqrt(cov[i][i]);
        }
        return sig;
    }

    
    @Override
    public PointVectorValuePair optimize(OptimizationData... optData)
        throws TooManyEvaluationsException {
        // Set up base class and perform computation.
        return super.optimize(optData);
    }

    
    protected double[] computeResiduals(double[] objectiveValue) {
        final double[] target = getTarget();
        if (objectiveValue.length != target.length) {
            throw new DimensionMismatchException(target.length,
                                                 objectiveValue.length);
        }

        final double[] residuals = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            residuals[i] = target[i] - objectiveValue[i];
        }

        return residuals;
    }

    
    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        // Allow base class to register its own data.
        super.parseOptimizationData(optData);

        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof Weight) {
                weightMatrixSqrt = squareRoot(((Weight) data).getWeight());
                // If more data must be parsed, this statement _must_ be
                // changed to "continue".
                break;
            }
        }
    }

    
    private RealMatrix squareRoot(RealMatrix m) {
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
}
