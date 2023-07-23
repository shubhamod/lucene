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

import org.apache.lucene.util.hnsw.math.analysis.DifferentiableMultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.analysis.FunctionUtils;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.MultivariateDifferentiableVectorFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.DiagonalMatrix;
import org.apache.lucene.util.hnsw.math.linear.DecompositionSolver;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.QRDecomposition;
import org.apache.lucene.util.hnsw.math.linear.EigenDecomposition;
import org.apache.lucene.util.hnsw.math.optimization.OptimizationData;
import org.apache.lucene.util.hnsw.math.optimization.InitialGuess;
import org.apache.lucene.util.hnsw.math.optimization.Target;
import org.apache.lucene.util.hnsw.math.optimization.Weight;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optimization.DifferentiableMultivariateVectorOptimizer;
import org.apache.lucene.util.hnsw.math.optimization.PointVectorValuePair;
import org.apache.lucene.util.hnsw.math.optimization.direct.BaseAbstractMultivariateVectorOptimizer;
import org.apache.lucene.util.hnsw.math.util.FastMath;


@Deprecated
public abstract class AbstractLeastSquaresOptimizer
    extends BaseAbstractMultivariateVectorOptimizer<DifferentiableMultivariateVectorFunction>
    implements DifferentiableMultivariateVectorOptimizer {
    
    @Deprecated
    private static final double DEFAULT_SINGULARITY_THRESHOLD = 1e-14;
    
    @Deprecated
    protected double[][] weightedResidualJacobian;
    
    @Deprecated
    protected int cols;
    
    @Deprecated
    protected int rows;
    
    @Deprecated
    protected double[] point;
    
    @Deprecated
    protected double[] objective;
    
    @Deprecated
    protected double[] weightedResiduals;
    
    @Deprecated
    protected double cost;
    
    private MultivariateDifferentiableVectorFunction jF;
    
    private int jacobianEvaluations;
    
    private RealMatrix weightMatrixSqrt;

    
    @Deprecated
    protected AbstractLeastSquaresOptimizer() {}

    
    protected AbstractLeastSquaresOptimizer(ConvergenceChecker<PointVectorValuePair> checker) {
        super(checker);
    }

    
    public int getJacobianEvaluations() {
        return jacobianEvaluations;
    }

    
    @Deprecated
    protected void updateJacobian() {
        final RealMatrix weightedJacobian = computeWeightedJacobian(point);
        weightedResidualJacobian = weightedJacobian.scalarMultiply(-1).getData();
    }

    
    protected RealMatrix computeWeightedJacobian(double[] params) {
        ++jacobianEvaluations;

        final DerivativeStructure[] dsPoint = new DerivativeStructure[params.length];
        final int nC = params.length;
        for (int i = 0; i < nC; ++i) {
            dsPoint[i] = new DerivativeStructure(nC, 1, i, params[i]);
        }
        final DerivativeStructure[] dsValue = jF.value(dsPoint);
        final int nR = getTarget().length;
        if (dsValue.length != nR) {
            throw new DimensionMismatchException(dsValue.length, nR);
        }
        final double[][] jacobianData = new double[nR][nC];
        for (int i = 0; i < nR; ++i) {
            int[] orders = new int[nC];
            for (int j = 0; j < nC; ++j) {
                orders[j] = 1;
                jacobianData[i][j] = dsValue[i].getPartialDerivative(orders);
                orders[j] = 0;
            }
        }

        return weightMatrixSqrt.multiply(MatrixUtils.createRealMatrix(jacobianData));
    }

    
    @Deprecated
    protected void updateResidualsAndCost() {
        objective = computeObjectiveValue(point);
        final double[] res = computeResiduals(objective);

        // Compute cost.
        cost = computeCost(res);

        // Compute weighted residuals.
        final ArrayRealVector residuals = new ArrayRealVector(res);
        weightedResiduals = weightMatrixSqrt.operate(residuals).toArray();
    }

    
    protected double computeCost(double[] residuals) {
        final ArrayRealVector r = new ArrayRealVector(residuals);
        return FastMath.sqrt(r.dotProduct(getWeight().operate(r)));
    }

    
    public double getRMS() {
        return FastMath.sqrt(getChiSquare() / rows);
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

    
    @Deprecated
    public double[][] getCovariances() {
        return getCovariances(DEFAULT_SINGULARITY_THRESHOLD);
    }

    
    @Deprecated
    public double[][] getCovariances(double threshold) {
        return computeCovariances(point, threshold);
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

    
    @Deprecated
    public double[] guessParametersErrors() {
        if (rows <= cols) {
            throw new NumberIsTooSmallException(LocalizedFormats.NO_DEGREES_OF_FREEDOM,
                                                rows, cols, false);
        }
        double[] errors = new double[cols];
        final double c = FastMath.sqrt(getChiSquare() / (rows - cols));
        double[][] covar = computeCovariances(point, 1e-14);
        for (int i = 0; i < errors.length; ++i) {
            errors[i] = FastMath.sqrt(covar[i][i]) * c;
        }
        return errors;
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
    @Deprecated
    public PointVectorValuePair optimize(int maxEval,
                                         final DifferentiableMultivariateVectorFunction f,
                                         final double[] target, final double[] weights,
                                         final double[] startPoint) {
        return optimizeInternal(maxEval,
                                FunctionUtils.toMultivariateDifferentiableVectorFunction(f),
                                new Target(target),
                                new Weight(weights),
                                new InitialGuess(startPoint));
    }

    
    @Deprecated
    public PointVectorValuePair optimize(final int maxEval,
                                         final MultivariateDifferentiableVectorFunction f,
                                         final double[] target, final double[] weights,
                                         final double[] startPoint) {
        return optimizeInternal(maxEval, f,
                                new Target(target),
                                new Weight(weights),
                                new InitialGuess(startPoint));
    }

    
    @Deprecated
    protected PointVectorValuePair optimizeInternal(final int maxEval,
                                                    final MultivariateDifferentiableVectorFunction f,
                                                    OptimizationData... optData) {
        // XXX Conversion will be removed when the generic argument of the
        // base class becomes "MultivariateDifferentiableVectorFunction".
        return super.optimizeInternal(maxEval, FunctionUtils.toDifferentiableMultivariateVectorFunction(f), optData);
    }

    
    @Override
    protected void setUp() {
        super.setUp();

        // Reset counter.
        jacobianEvaluations = 0;

        // Square-root of the weight matrix.
        weightMatrixSqrt = squareRoot(getWeight());

        // Store least squares problem characteristics.
        // XXX The conversion won't be necessary when the generic argument of
        // the base class becomes "MultivariateDifferentiableVectorFunction".
        // XXX "jF" is not strictly necessary anymore but is currently more
        // efficient than converting the value returned from "getObjectiveFunction()"
        // every time it is used.
        jF = FunctionUtils.toMultivariateDifferentiableVectorFunction((DifferentiableMultivariateVectorFunction) getObjectiveFunction());

        // Arrays shared with "private" and "protected" methods.
        point = getStartPoint();
        rows = getTarget().length;
        cols = point.length;
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
