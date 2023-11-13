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

import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem.Evaluation;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;


class OptimumImpl implements Optimum {

    
    private final Evaluation value;
    
    private final int evaluations;
    
    private final int iterations;

    
    OptimumImpl(final Evaluation value, final int evaluations, final int iterations) {
        this.value = value;
        this.evaluations = evaluations;
        this.iterations = iterations;
    }

    /* auto-generated implementations */

    
    public int getEvaluations() {
        return evaluations;
    }

    
    public int getIterations() {
        return iterations;
    }

    
    public RealMatrix getCovariances(double threshold) {
        return value.getCovariances(threshold);
    }

    
    public RealVector getSigma(double covarianceSingularityThreshold) {
        return value.getSigma(covarianceSingularityThreshold);
    }

    
    public double getRMS() {
        return value.getRMS();
    }

    
    public RealMatrix getJacobian() {
        return value.getJacobian();
    }

    
    public double getCost() {
        return value.getCost();
    }

    
    public RealVector getResiduals() {
        return value.getResiduals();
    }

    
    public RealVector getPoint() {
        return value.getPoint();
    }
}
