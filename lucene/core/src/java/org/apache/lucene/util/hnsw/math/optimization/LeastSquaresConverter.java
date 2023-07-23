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

package org.apache.lucene.util.hnsw.math.optimization;

import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;



@Deprecated
public class LeastSquaresConverter implements MultivariateFunction {

    
    private final MultivariateVectorFunction function;

    
    private final double[] observations;

    
    private final double[] weights;

    
    private final RealMatrix scale;

    
    public LeastSquaresConverter(final MultivariateVectorFunction function,
                                 final double[] observations) {
        this.function     = function;
        this.observations = observations.clone();
        this.weights      = null;
        this.scale        = null;
    }

    
    public LeastSquaresConverter(final MultivariateVectorFunction function,
                                 final double[] observations, final double[] weights) {
        if (observations.length != weights.length) {
            throw new DimensionMismatchException(observations.length, weights.length);
        }
        this.function     = function;
        this.observations = observations.clone();
        this.weights      = weights.clone();
        this.scale        = null;
    }

    
    public LeastSquaresConverter(final MultivariateVectorFunction function,
                                 final double[] observations, final RealMatrix scale) {
        if (observations.length != scale.getColumnDimension()) {
            throw new DimensionMismatchException(observations.length, scale.getColumnDimension());
        }
        this.function     = function;
        this.observations = observations.clone();
        this.weights      = null;
        this.scale        = scale.copy();
    }

    
    public double value(final double[] point) {
        // compute residuals
        final double[] residuals = function.value(point);
        if (residuals.length != observations.length) {
            throw new DimensionMismatchException(residuals.length, observations.length);
        }
        for (int i = 0; i < residuals.length; ++i) {
            residuals[i] -= observations[i];
        }

        // compute sum of squares
        double sumSquares = 0;
        if (weights != null) {
            for (int i = 0; i < residuals.length; ++i) {
                final double ri = residuals[i];
                sumSquares +=  weights[i] * ri * ri;
            }
        } else if (scale != null) {
            for (final double yi : scale.operate(residuals)) {
                sumSquares += yi * yi;
            }
        } else {
            for (final double ri : residuals) {
                sumSquares += ri * ri;
            }
        }

        return sumSquares;
    }
}
