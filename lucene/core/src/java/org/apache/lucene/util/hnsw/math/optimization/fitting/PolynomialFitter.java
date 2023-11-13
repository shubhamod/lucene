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

import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialFunction;
import org.apache.lucene.util.hnsw.math.optimization.DifferentiableMultivariateVectorOptimizer;


@Deprecated
public class PolynomialFitter extends CurveFitter<PolynomialFunction.Parametric> {
    
    @Deprecated
    private final int degree;

    
    @Deprecated
    public PolynomialFitter(int degree, final DifferentiableMultivariateVectorOptimizer optimizer) {
        super(optimizer);
        this.degree = degree;
    }

    
    public PolynomialFitter(DifferentiableMultivariateVectorOptimizer optimizer) {
        super(optimizer);
        degree = -1; // To avoid compilation error until the instance variable is removed.
    }

    
    @Deprecated
    public double[] fit() {
        return fit(new PolynomialFunction.Parametric(), new double[degree + 1]);
    }

    
    public double[] fit(int maxEval, double[] guess) {
        return fit(maxEval, new PolynomialFunction.Parametric(), guess);
    }

    
    public double[] fit(double[] guess) {
        return fit(new PolynomialFunction.Parametric(), guess);
    }
}
