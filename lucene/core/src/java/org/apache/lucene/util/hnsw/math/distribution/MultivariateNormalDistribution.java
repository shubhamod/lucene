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
package org.apache.lucene.util.hnsw.math.distribution;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.EigenDecomposition;
import org.apache.lucene.util.hnsw.math.linear.NonPositiveDefiniteMatrixException;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.SingularMatrixException;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class MultivariateNormalDistribution
    extends AbstractMultivariateRealDistribution {
    
    private final double[] means;
    
    private final RealMatrix covarianceMatrix;
    
    private final RealMatrix covarianceMatrixInverse;
    
    private final double covarianceMatrixDeterminant;
    
    private final RealMatrix samplingMatrix;

    
    public MultivariateNormalDistribution(final double[] means,
                                          final double[][] covariances)
        throws SingularMatrixException,
               DimensionMismatchException,
               NonPositiveDefiniteMatrixException {
        this(new Well19937c(), means, covariances);
    }

    
    public MultivariateNormalDistribution(RandomGenerator rng,
                                          final double[] means,
                                          final double[][] covariances)
            throws SingularMatrixException,
                   DimensionMismatchException,
                   NonPositiveDefiniteMatrixException {
        super(rng, means.length);

        final int dim = means.length;

        if (covariances.length != dim) {
            throw new DimensionMismatchException(covariances.length, dim);
        }

        for (int i = 0; i < dim; i++) {
            if (dim != covariances[i].length) {
                throw new DimensionMismatchException(covariances[i].length, dim);
            }
        }

        this.means = MathArrays.copyOf(means);

        covarianceMatrix = new Array2DRowRealMatrix(covariances);

        // Covariance matrix eigen decomposition.
        final EigenDecomposition covMatDec = new EigenDecomposition(covarianceMatrix);

        // Compute and store the inverse.
        covarianceMatrixInverse = covMatDec.getSolver().getInverse();
        // Compute and store the determinant.
        covarianceMatrixDeterminant = covMatDec.getDeterminant();

        // Eigenvalues of the covariance matrix.
        final double[] covMatEigenvalues = covMatDec.getRealEigenvalues();

        for (int i = 0; i < covMatEigenvalues.length; i++) {
            if (covMatEigenvalues[i] < 0) {
                throw new NonPositiveDefiniteMatrixException(covMatEigenvalues[i], i, 0);
            }
        }

        // Matrix where each column is an eigenvector of the covariance matrix.
        final Array2DRowRealMatrix covMatEigenvectors = new Array2DRowRealMatrix(dim, dim);
        for (int v = 0; v < dim; v++) {
            final double[] evec = covMatDec.getEigenvector(v).toArray();
            covMatEigenvectors.setColumn(v, evec);
        }

        final RealMatrix tmpMatrix = covMatEigenvectors.transpose();

        // Scale each eigenvector by the square root of its eigenvalue.
        for (int row = 0; row < dim; row++) {
            final double factor = FastMath.sqrt(covMatEigenvalues[row]);
            for (int col = 0; col < dim; col++) {
                tmpMatrix.multiplyEntry(row, col, factor);
            }
        }

        samplingMatrix = covMatEigenvectors.multiply(tmpMatrix);
    }

    
    public double[] getMeans() {
        return MathArrays.copyOf(means);
    }

    
    public RealMatrix getCovariances() {
        return covarianceMatrix.copy();
    }

    
    public double density(final double[] vals) throws DimensionMismatchException {
        final int dim = getDimension();
        if (vals.length != dim) {
            throw new DimensionMismatchException(vals.length, dim);
        }

        return FastMath.pow(2 * FastMath.PI, -0.5 * dim) *
            FastMath.pow(covarianceMatrixDeterminant, -0.5) *
            getExponentTerm(vals);
    }

    
    public double[] getStandardDeviations() {
        final int dim = getDimension();
        final double[] std = new double[dim];
        final double[][] s = covarianceMatrix.getData();
        for (int i = 0; i < dim; i++) {
            std[i] = FastMath.sqrt(s[i][i]);
        }
        return std;
    }

    
    @Override
    public double[] sample() {
        final int dim = getDimension();
        final double[] normalVals = new double[dim];

        for (int i = 0; i < dim; i++) {
            normalVals[i] = random.nextGaussian();
        }

        final double[] vals = samplingMatrix.operate(normalVals);

        for (int i = 0; i < dim; i++) {
            vals[i] += means[i];
        }

        return vals;
    }

    
    private double getExponentTerm(final double[] values) {
        final double[] centered = new double[values.length];
        for (int i = 0; i < centered.length; i++) {
            centered[i] = values[i] - getMeans()[i];
        }
        final double[] preMultiplied = covarianceMatrixInverse.preMultiply(centered);
        double sum = 0;
        for (int i = 0; i < preMultiplied.length; i++) {
            sum += preMultiplied[i] * centered[i];
        }
        return FastMath.exp(-0.5 * sum);
    }
}
