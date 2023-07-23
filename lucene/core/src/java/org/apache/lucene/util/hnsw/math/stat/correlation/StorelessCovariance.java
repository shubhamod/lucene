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
package org.apache.lucene.util.hnsw.math.stat.correlation;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathUnsupportedOperationException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;


public class StorelessCovariance extends Covariance {

    
    private StorelessBivariateCovariance[] covMatrix;

    
    private int dimension;

    
    public StorelessCovariance(final int dim) {
        this(dim, true);
    }

    
    public StorelessCovariance(final int dim, final boolean biasCorrected) {
        dimension = dim;
        covMatrix = new StorelessBivariateCovariance[dimension * (dimension + 1) / 2];
        initializeMatrix(biasCorrected);
    }

    
    private void initializeMatrix(final boolean biasCorrected) {
        for(int i = 0; i < dimension; i++){
            for(int j = 0; j < dimension; j++){
                setElement(i, j, new StorelessBivariateCovariance(biasCorrected));
            }
        }
    }

    
    private int indexOf(final int i, final int j) {
        return j < i ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
    }

    
    private StorelessBivariateCovariance getElement(final int i, final int j) {
        return covMatrix[indexOf(i, j)];
    }

    
    private void setElement(final int i, final int j,
                            final StorelessBivariateCovariance cov) {
        covMatrix[indexOf(i, j)] = cov;
    }

    
    public double getCovariance(final int xIndex,
                                final int yIndex)
        throws NumberIsTooSmallException {

        return getElement(xIndex, yIndex).getResult();

    }

    
    public void increment(final double[] data)
        throws DimensionMismatchException {

        int length = data.length;
        if (length != dimension) {
            throw new DimensionMismatchException(length, dimension);
        }

        // only update the upper triangular part of the covariance matrix
        // as only these parts are actually stored
        for (int i = 0; i < length; i++){
            for (int j = i; j < length; j++){
                getElement(i, j).increment(data[i], data[j]);
            }
        }

    }

    
    public void append(StorelessCovariance sc) throws DimensionMismatchException {
        if (sc.dimension != dimension) {
            throw new DimensionMismatchException(sc.dimension, dimension);
        }

        // only update the upper triangular part of the covariance matrix
        // as only these parts are actually stored
        for (int i = 0; i < dimension; i++) {
            for (int j = i; j < dimension; j++) {
                getElement(i, j).append(sc.getElement(i, j));
            }
        }
    }

    
    @Override
    public RealMatrix getCovarianceMatrix() throws NumberIsTooSmallException {
        return MatrixUtils.createRealMatrix(getData());
    }

    
    public double[][] getData() throws NumberIsTooSmallException {
        final double[][] data = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                data[i][j] = getElement(i, j).getResult();
            }
        }
        return data;
    }

    
    @Override
    public int getN()
        throws MathUnsupportedOperationException {
        throw new MathUnsupportedOperationException();
    }
}
