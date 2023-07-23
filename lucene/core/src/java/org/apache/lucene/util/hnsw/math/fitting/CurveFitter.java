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

import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateMatrixFunction;
import org.apache.lucene.util.hnsw.math.analysis.ParametricUnivariateFunction;
import org.apache.lucene.util.hnsw.math.optim.MaxEval;
import org.apache.lucene.util.hnsw.math.optim.InitialGuess;
import org.apache.lucene.util.hnsw.math.optim.PointVectorValuePair;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.MultivariateVectorOptimizer;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.ModelFunction;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.ModelFunctionJacobian;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.Target;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.vector.Weight;


@Deprecated
public class CurveFitter<T extends ParametricUnivariateFunction> {
    
    private final MultivariateVectorOptimizer optimizer;
    
    private final List<WeightedObservedPoint> observations;

    
    public CurveFitter(final MultivariateVectorOptimizer optimizer) {
        this.optimizer = optimizer;
        observations = new ArrayList<WeightedObservedPoint>();
    }

    
    public void addObservedPoint(double x, double y) {
        addObservedPoint(1.0, x, y);
    }

    
    public void addObservedPoint(double weight, double x, double y) {
        observations.add(new WeightedObservedPoint(weight, x, y));
    }

    
    public void addObservedPoint(WeightedObservedPoint observed) {
        observations.add(observed);
    }

    
    public WeightedObservedPoint[] getObservations() {
        return observations.toArray(new WeightedObservedPoint[observations.size()]);
    }

    
    public void clearObservations() {
        observations.clear();
    }

    
    public double[] fit(T f, final double[] initialGuess) {
        return fit(Integer.MAX_VALUE, f, initialGuess);
    }

    
    public double[] fit(int maxEval, T f,
                        final double[] initialGuess) {
        // Prepare least squares problem.
        double[] target  = new double[observations.size()];
        double[] weights = new double[observations.size()];
        int i = 0;
        for (WeightedObservedPoint point : observations) {
            target[i]  = point.getY();
            weights[i] = point.getWeight();
            ++i;
        }

        // Input to the optimizer: the model and its Jacobian.
        final TheoreticalValuesFunction model = new TheoreticalValuesFunction(f);

        // Perform the fit.
        final PointVectorValuePair optimum
            = optimizer.optimize(new MaxEval(maxEval),
                                 model.getModelFunction(),
                                 model.getModelFunctionJacobian(),
                                 new Target(target),
                                 new Weight(weights),
                                 new InitialGuess(initialGuess));
        // Extract the coefficients.
        return optimum.getPointRef();
    }

    
    private class TheoreticalValuesFunction {
        
        private final ParametricUnivariateFunction f;

        
        TheoreticalValuesFunction(final ParametricUnivariateFunction f) {
            this.f = f;
        }

        
        public ModelFunction getModelFunction() {
            return new ModelFunction(new MultivariateVectorFunction() {
                    
                    public double[] value(double[] point) {
                        // compute the residuals
                        final double[] values = new double[observations.size()];
                        int i = 0;
                        for (WeightedObservedPoint observed : observations) {
                            values[i++] = f.value(observed.getX(), point);
                        }

                        return values;
                    }
                });
        }

        
        public ModelFunctionJacobian getModelFunctionJacobian() {
            return new ModelFunctionJacobian(new MultivariateMatrixFunction() {
                    
                    public double[][] value(double[] point) {
                        final double[][] jacobian = new double[observations.size()][];
                        int i = 0;
                        for (WeightedObservedPoint observed : observations) {
                            jacobian[i++] = f.gradient(observed.getX(), point);
                        }
                        return jacobian;
                    }
                });
        }
    }
}
