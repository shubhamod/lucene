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

package org.apache.lucene.util.hnsw.math.random;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RectangularCholeskyDecomposition;



public class CorrelatedRandomVectorGenerator
    implements RandomVectorGenerator {
    
    private final double[] mean;
    
    private final NormalizedRandomGenerator generator;
    
    private final double[] normalized;
    
    private final RealMatrix root;

    
    public CorrelatedRandomVectorGenerator(double[] mean,
                                           RealMatrix covariance, double small,
                                           NormalizedRandomGenerator generator) {
        int order = covariance.getRowDimension();
        if (mean.length != order) {
            throw new DimensionMismatchException(mean.length, order);
        }
        this.mean = mean.clone();

        final RectangularCholeskyDecomposition decomposition =
            new RectangularCholeskyDecomposition(covariance, small);
        root = decomposition.getRootMatrix();

        this.generator = generator;
        normalized = new double[decomposition.getRank()];

    }

    
    public CorrelatedRandomVectorGenerator(RealMatrix covariance, double small,
                                           NormalizedRandomGenerator generator) {
        int order = covariance.getRowDimension();
        mean = new double[order];
        for (int i = 0; i < order; ++i) {
            mean[i] = 0;
        }

        final RectangularCholeskyDecomposition decomposition =
            new RectangularCholeskyDecomposition(covariance, small);
        root = decomposition.getRootMatrix();

        this.generator = generator;
        normalized = new double[decomposition.getRank()];

    }

    
    public NormalizedRandomGenerator getGenerator() {
        return generator;
    }

    
    public int getRank() {
        return normalized.length;
    }

    
    public RealMatrix getRootMatrix() {
        return root;
    }

    
    public double[] nextVector() {

        // generate uncorrelated vector
        for (int i = 0; i < normalized.length; ++i) {
            normalized[i] = generator.nextNormalizedDouble();
        }

        // compute correlated vector
        double[] correlated = new double[mean.length];
        for (int i = 0; i < correlated.length; ++i) {
            correlated[i] = mean[i];
            for (int j = 0; j < root.getColumnDimension(); ++j) {
                correlated[i] += root.getEntry(i, j) * normalized[j];
            }
        }

        return correlated;

    }

}
