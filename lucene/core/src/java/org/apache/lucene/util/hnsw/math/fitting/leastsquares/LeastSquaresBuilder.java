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

import org.apache.lucene.util.hnsw.math.analysis.MultivariateMatrixFunction;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem.Evaluation;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optim.PointVectorValuePair;


public class LeastSquaresBuilder {

    
    private int maxEvaluations;
    
    private int maxIterations;
    
    private ConvergenceChecker<Evaluation> checker;
    
    private MultivariateJacobianFunction model;
    
    private RealVector target;
    
    private RealVector start;
    
    private RealMatrix weight;
    
    private boolean lazyEvaluation;
    
    private ParameterValidator paramValidator;


    
    public LeastSquaresProblem build() {
        return LeastSquaresFactory.create(model,
                                          target,
                                          start,
                                          weight,
                                          checker,
                                          maxEvaluations,
                                          maxIterations,
                                          lazyEvaluation,
                                          paramValidator);
    }

    
    public LeastSquaresBuilder maxEvaluations(final int newMaxEvaluations) {
        this.maxEvaluations = newMaxEvaluations;
        return this;
    }

    
    public LeastSquaresBuilder maxIterations(final int newMaxIterations) {
        this.maxIterations = newMaxIterations;
        return this;
    }

    
    public LeastSquaresBuilder checker(final ConvergenceChecker<Evaluation> newChecker) {
        this.checker = newChecker;
        return this;
    }

    
    public LeastSquaresBuilder checkerPair(final ConvergenceChecker<PointVectorValuePair> newChecker) {
        return this.checker(LeastSquaresFactory.evaluationChecker(newChecker));
    }

    
    public LeastSquaresBuilder model(final MultivariateVectorFunction value,
                                     final MultivariateMatrixFunction jacobian) {
        return model(LeastSquaresFactory.model(value, jacobian));
    }

    
    public LeastSquaresBuilder model(final MultivariateJacobianFunction newModel) {
        this.model = newModel;
        return this;
    }

    
    public LeastSquaresBuilder target(final RealVector newTarget) {
        this.target = newTarget;
        return this;
    }

    
    public LeastSquaresBuilder target(final double[] newTarget) {
        return target(new ArrayRealVector(newTarget, false));
    }

    
    public LeastSquaresBuilder start(final RealVector newStart) {
        this.start = newStart;
        return this;
    }

    
    public LeastSquaresBuilder start(final double[] newStart) {
        return start(new ArrayRealVector(newStart, false));
    }

    
    public LeastSquaresBuilder weight(final RealMatrix newWeight) {
        this.weight = newWeight;
        return this;
    }

    
    public LeastSquaresBuilder lazyEvaluation(final boolean newValue) {
        lazyEvaluation = newValue;
        return this;
    }

    
    public LeastSquaresBuilder parameterValidator(final ParameterValidator newValidator) {
        paramValidator = newValidator;
        return this;
    }
}
