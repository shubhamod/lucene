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

import org.apache.lucene.util.hnsw.math.analysis.MultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateMatrixFunction;
import org.apache.lucene.util.hnsw.math.analysis.ParametricUnivariateFunction;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LevenbergMarquardtOptimizer;


public abstract class AbstractCurveFitter {
    
    public double[] fit(Collection<WeightedObservedPoint> points) {
        // Perform the fit.
        return getOptimizer().optimize(getProblem(points)).getPoint().toArray();
    }

    
    protected LeastSquaresOptimizer getOptimizer() {
        return new LevenbergMarquardtOptimizer();
    }

    
    protected abstract LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> points);

    
    protected static class TheoreticalValuesFunction {
        
        private final ParametricUnivariateFunction f;
        
        private final double[] points;

        
        public TheoreticalValuesFunction(final ParametricUnivariateFunction f,
                                         final Collection<WeightedObservedPoint> observations) {
            this.f = f;

            final int len = observations.size();
            this.points = new double[len];
            int i = 0;
            for (WeightedObservedPoint obs : observations) {
                this.points[i++] = obs.getX();
            }
        }

        
        public MultivariateVectorFunction getModelFunction() {
            return new MultivariateVectorFunction() {
                
                public double[] value(double[] p) {
                    final int len = points.length;
                    final double[] values = new double[len];
                    for (int i = 0; i < len; i++) {
                        values[i] = f.value(points[i], p);
                    }

                    return values;
                }
            };
        }

        
        public MultivariateMatrixFunction getModelFunctionJacobian() {
            return new MultivariateMatrixFunction() {
                
                public double[][] value(double[] p) {
                    final int len = points.length;
                    final double[][] jacobian = new double[len][];
                    for (int i = 0; i < len; i++) {
                        jacobian[i] = f.gradient(points[i], p);
                    }
                    return jacobian;
                }
            };
        }
    }
}
