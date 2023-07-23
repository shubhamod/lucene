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
package org.apache.lucene.util.hnsw.math.fitting;

import java.util.Collection;

import org.apache.lucene.util.hnsw.math.analysis.ParametricUnivariateFunction;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem;
import org.apache.lucene.util.hnsw.math.linear.DiagonalMatrix;


public class SimpleCurveFitter extends AbstractCurveFitter {
    
    private final ParametricUnivariateFunction function;
    
    private final double[] initialGuess;
    
    private final int maxIter;

    
    private SimpleCurveFitter(ParametricUnivariateFunction function,
                              double[] initialGuess,
                              int maxIter) {
        this.function = function;
        this.initialGuess = initialGuess;
        this.maxIter = maxIter;
    }

    
    public static SimpleCurveFitter create(ParametricUnivariateFunction f,
                                           double[] start) {
        return new SimpleCurveFitter(f, start, Integer.MAX_VALUE);
    }

    
    public SimpleCurveFitter withStartPoint(double[] newStart) {
        return new SimpleCurveFitter(function,
                                     newStart.clone(),
                                     maxIter);
    }

    
    public SimpleCurveFitter withMaxIterations(int newMaxIter) {
        return new SimpleCurveFitter(function,
                                     initialGuess,
                                     newMaxIter);
    }

    
    @Override
    protected LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> observations) {
        // Prepare least-squares problem.
        final int len = observations.size();
        final double[] target  = new double[len];
        final double[] weights = new double[len];

        int count = 0;
        for (WeightedObservedPoint obs : observations) {
            target[count]  = obs.getY();
            weights[count] = obs.getWeight();
            ++count;
        }

        final AbstractCurveFitter.TheoreticalValuesFunction model
            = new AbstractCurveFitter.TheoreticalValuesFunction(function,
                                                                observations);

        // Create an optimizer for fitting the curve to the observed points.
        return new LeastSquaresBuilder().
                maxEvaluations(Integer.MAX_VALUE).
                maxIterations(maxIter).
                start(initialGuess).
                target(target).
                weight(new DiagonalMatrix(weights)).
                model(model.getModelFunction(), model.getModelFunctionJacobian()).
                build();
    }
}
