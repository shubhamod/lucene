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

package org.apache.lucene.util.hnsw.math.analysis.function;

import org.apache.lucene.util.hnsw.math.analysis.DifferentiableUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.FunctionUtils;
import org.apache.lucene.util.hnsw.math.analysis.ParametricUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class HarmonicOscillator implements UnivariateDifferentiableFunction, DifferentiableUnivariateFunction {
    
    private final double amplitude;
    
    private final double omega;
    
    private final double phase;

    
    public HarmonicOscillator(double amplitude,
                              double omega,
                              double phase) {
        this.amplitude = amplitude;
        this.omega = omega;
        this.phase = phase;
    }

    
    public double value(double x) {
        return value(omega * x + phase, amplitude);
    }

    
    @Deprecated
    public UnivariateFunction derivative() {
        return FunctionUtils.toDifferentiableUnivariateFunction(this).derivative();
    }

    
    public static class Parametric implements ParametricUnivariateFunction {
        
        public double value(double x, double ... param)
            throws NullArgumentException,
                   DimensionMismatchException {
            validateParameters(param);
            return HarmonicOscillator.value(x * param[1] + param[2], param[0]);
        }

        
        public double[] gradient(double x, double ... param)
            throws NullArgumentException,
                   DimensionMismatchException {
            validateParameters(param);

            final double amplitude = param[0];
            final double omega = param[1];
            final double phase = param[2];

            final double xTimesOmegaPlusPhase = omega * x + phase;
            final double a = HarmonicOscillator.value(xTimesOmegaPlusPhase, 1);
            final double p = -amplitude * FastMath.sin(xTimesOmegaPlusPhase);
            final double w = p * x;

            return new double[] { a, w, p };
        }

        
        private void validateParameters(double[] param)
            throws NullArgumentException,
                   DimensionMismatchException {
            if (param == null) {
                throw new NullArgumentException();
            }
            if (param.length != 3) {
                throw new DimensionMismatchException(param.length, 3);
            }
        }
    }

    
    private static double value(double xTimesOmegaPlusPhase,
                                double amplitude) {
        return amplitude * FastMath.cos(xTimesOmegaPlusPhase);
    }

    
    public DerivativeStructure value(final DerivativeStructure t)
        throws DimensionMismatchException {
        final double x = t.getValue();
        double[] f = new double[t.getOrder() + 1];

        final double alpha = omega * x + phase;
        f[0] = amplitude * FastMath.cos(alpha);
        if (f.length > 1) {
            f[1] = -amplitude * omega * FastMath.sin(alpha);
            final double mo2 = - omega * omega;
            for (int i = 2; i < f.length; ++i) {
                f[i] = mo2 * f[i - 2];
            }
        }

        return t.compose(f);

    }

}
