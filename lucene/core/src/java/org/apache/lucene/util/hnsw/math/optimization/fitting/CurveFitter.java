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

package org.apache.lucene.util.hnsw.math.optimization.fitting;

import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.util.hnsw.math.analysis.DifferentiableMultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateMatrixFunction;
import org.apache.lucene.util.hnsw.math.analysis.ParametricUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.MultivariateDifferentiableVectorFunction;
import org.apache.lucene.util.hnsw.math.optimization.DifferentiableMultivariateVectorOptimizer;
import org.apache.lucene.util.hnsw.math.optimization.MultivariateDifferentiableVectorOptimizer;
import org.apache.lucene.util.hnsw.math.optimization.PointVectorValuePair;


@Deprecated
public class CurveFitter<T extends ParametricUnivariateFunction> {

    
    @Deprecated
    private final DifferentiableMultivariateVectorOptimizer oldOptimizer;

    
    private final MultivariateDifferentiableVectorOptimizer optimizer;

    
    private final List<WeightedObservedPoint> observations;

    
    @Deprecated
    public CurveFitter(final DifferentiableMultivariateVectorOptimizer optimizer) {
        this.oldOptimizer = optimizer;
        this.optimizer    = null;
        observations      = new ArrayList<WeightedObservedPoint>();
    }

    
    public CurveFitter(final MultivariateDifferentiableVectorOptimizer optimizer) {
        this.oldOptimizer = null;
        this.optimizer    = optimizer;
        observations      = new ArrayList<WeightedObservedPoint>();
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
        // prepare least squares problem
        double[] target  = new double[observations.size()];
        double[] weights = new double[observations.size()];
        int i = 0;
        for (WeightedObservedPoint point : observations) {
            target[i]  = point.getY();
            weights[i] = point.getWeight();
            ++i;
        }

        // perform the fit
        final PointVectorValuePair optimum;
        if (optimizer == null) {
            // to be removed in 4.0
            optimum = oldOptimizer.optimize(maxEval, new OldTheoreticalValuesFunction(f),
                                            target, weights, initialGuess);
        } else {
            optimum = optimizer.optimize(maxEval, new TheoreticalValuesFunction(f),
                                         target, weights, initialGuess);
        }

        // extract the coefficients
        return optimum.getPointRef();
    }

    
    @Deprecated
    private class OldTheoreticalValuesFunction
        implements DifferentiableMultivariateVectorFunction {
        
        private final ParametricUnivariateFunction f;

        
        OldTheoreticalValuesFunction(final ParametricUnivariateFunction f) {
            this.f = f;
        }

        
        public MultivariateMatrixFunction jacobian() {
            return new MultivariateMatrixFunction() {
                
                public double[][] value(double[] point) {
                    final double[][] jacobian = new double[observations.size()][];

                    int i = 0;
                    for (WeightedObservedPoint observed : observations) {
                        jacobian[i++] = f.gradient(observed.getX(), point);
                    }

                    return jacobian;
                }
            };
        }

        
        public double[] value(double[] point) {
            // compute the residuals
            final double[] values = new double[observations.size()];
            int i = 0;
            for (WeightedObservedPoint observed : observations) {
                values[i++] = f.value(observed.getX(), point);
            }

            return values;
        }
    }

    
    private class TheoreticalValuesFunction implements MultivariateDifferentiableVectorFunction {

        
        private final ParametricUnivariateFunction f;

        
        TheoreticalValuesFunction(final ParametricUnivariateFunction f) {
            this.f = f;
        }

        
        public double[] value(double[] point) {
            // compute the residuals
            final double[] values = new double[observations.size()];
            int i = 0;
            for (WeightedObservedPoint observed : observations) {
                values[i++] = f.value(observed.getX(), point);
            }

            return values;
        }

        
        public DerivativeStructure[] value(DerivativeStructure[] point) {

            // extract parameters
            final double[] parameters = new double[point.length];
            for (int k = 0; k < point.length; ++k) {
                parameters[k] = point[k].getValue();
            }

            // compute the residuals
            final DerivativeStructure[] values = new DerivativeStructure[observations.size()];
            int i = 0;
            for (WeightedObservedPoint observed : observations) {

                // build the DerivativeStructure by adding first the value as a constant
                // and then adding derivatives
                DerivativeStructure vi = new DerivativeStructure(point.length, 1, f.value(observed.getX(), parameters));
                for (int k = 0; k < point.length; ++k) {
                    vi = vi.add(new DerivativeStructure(point.length, 1, k, 0.0));
                }

                values[i++] = vi;

            }

            return values;
        }

    }

}
