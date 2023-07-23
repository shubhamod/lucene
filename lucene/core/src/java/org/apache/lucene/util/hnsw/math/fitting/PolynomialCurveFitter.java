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

import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialFunction;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem;
import org.apache.lucene.util.hnsw.math.linear.DiagonalMatrix;


public class PolynomialCurveFitter extends AbstractCurveFitter {
    
    private static final PolynomialFunction.Parametric FUNCTION = new PolynomialFunction.Parametric();
    
    private final double[] initialGuess;
    
    private final int maxIter;

    
    private PolynomialCurveFitter(double[] initialGuess,
                                  int maxIter) {
        this.initialGuess = initialGuess;
        this.maxIter = maxIter;
    }

    
    public static PolynomialCurveFitter create(int degree) {
        return new PolynomialCurveFitter(new double[degree + 1], Integer.MAX_VALUE);
    }

    
    public PolynomialCurveFitter withStartPoint(double[] newStart) {
        return new PolynomialCurveFitter(newStart.clone(),
                                         maxIter);
    }

    
    public PolynomialCurveFitter withMaxIterations(int newMaxIter) {
        return new PolynomialCurveFitter(initialGuess,
                                         newMaxIter);
    }

    
    @Override
    protected LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> observations) {
        // Prepare least-squares problem.
        final int len = observations.size();
        final double[] target  = new double[len];
        final double[] weights = new double[len];

        int i = 0;
        for (WeightedObservedPoint obs : observations) {
            target[i]  = obs.getY();
            weights[i] = obs.getWeight();
            ++i;
        }

        final AbstractCurveFitter.TheoreticalValuesFunction model =
                new AbstractCurveFitter.TheoreticalValuesFunction(FUNCTION, observations);

        if (initialGuess == null) {
            throw new MathInternalError();
        }

        // Return a new least squares problem set up to fit a polynomial curve to the
        // observed points.
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
