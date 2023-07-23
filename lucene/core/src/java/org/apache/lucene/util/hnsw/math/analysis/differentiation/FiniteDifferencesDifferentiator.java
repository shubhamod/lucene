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
package org.apache.lucene.util.hnsw.math.analysis.differentiation;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateMatrixFunction;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class FiniteDifferencesDifferentiator
    implements UnivariateFunctionDifferentiator, UnivariateVectorFunctionDifferentiator,
               UnivariateMatrixFunctionDifferentiator, Serializable {

    
    private static final long serialVersionUID = 20120917L;

    
    private final int nbPoints;

    
    private final double stepSize;

    
    private final double halfSampleSpan;

    
    private final double tMin;

    
    private final double tMax;

    
    public FiniteDifferencesDifferentiator(final int nbPoints, final double stepSize)
        throws NotPositiveException, NumberIsTooSmallException {
        this(nbPoints, stepSize, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    
    public FiniteDifferencesDifferentiator(final int nbPoints, final double stepSize,
                                           final double tLower, final double tUpper)
            throws NotPositiveException, NumberIsTooSmallException, NumberIsTooLargeException {

        if (nbPoints <= 1) {
            throw new NumberIsTooSmallException(stepSize, 1, false);
        }
        this.nbPoints = nbPoints;

        if (stepSize <= 0) {
            throw new NotPositiveException(stepSize);
        }
        this.stepSize = stepSize;

        halfSampleSpan = 0.5 * stepSize * (nbPoints - 1);
        if (2 * halfSampleSpan >= tUpper - tLower) {
            throw new NumberIsTooLargeException(2 * halfSampleSpan, tUpper - tLower, false);
        }
        final double safety = FastMath.ulp(halfSampleSpan);
        this.tMin = tLower + halfSampleSpan + safety;
        this.tMax = tUpper - halfSampleSpan - safety;

    }

    
    public int getNbPoints() {
        return nbPoints;
    }

    
    public double getStepSize() {
        return stepSize;
    }

    
    private DerivativeStructure evaluate(final DerivativeStructure t, final double t0,
                                         final double[] y)
        throws NumberIsTooLargeException {

        // create divided differences diagonal arrays
        final double[] top    = new double[nbPoints];
        final double[] bottom = new double[nbPoints];

        for (int i = 0; i < nbPoints; ++i) {

            // update the bottom diagonal of the divided differences array
            bottom[i] = y[i];
            for (int j = 1; j <= i; ++j) {
                bottom[i - j] = (bottom[i - j + 1] - bottom[i - j]) / (j * stepSize);
            }

            // update the top diagonal of the divided differences array
            top[i] = bottom[0];

        }

        // evaluate interpolation polynomial (represented by top diagonal) at t
        final int order            = t.getOrder();
        final int parameters       = t.getFreeParameters();
        final double[] derivatives = t.getAllDerivatives();
        final double dt0           = t.getValue() - t0;
        DerivativeStructure interpolation = new DerivativeStructure(parameters, order, 0.0);
        DerivativeStructure monomial = null;
        for (int i = 0; i < nbPoints; ++i) {
            if (i == 0) {
                // start with monomial(t) = 1
                monomial = new DerivativeStructure(parameters, order, 1.0);
            } else {
                // monomial(t) = (t - t0) * (t - t1) * ... * (t - t(i-1))
                derivatives[0] = dt0 - (i - 1) * stepSize;
                final DerivativeStructure deltaX = new DerivativeStructure(parameters, order, derivatives);
                monomial = monomial.multiply(deltaX);
            }
            interpolation = interpolation.add(monomial.multiply(top[i]));
        }

        return interpolation;

    }

    
    public UnivariateDifferentiableFunction differentiate(final UnivariateFunction function) {
        return new UnivariateDifferentiableFunction() {

            
            public double value(final double x) throws MathIllegalArgumentException {
                return function.value(x);
            }

            
            public DerivativeStructure value(final DerivativeStructure t)
                throws MathIllegalArgumentException {

                // check we can achieve the requested derivation order with the sample
                if (t.getOrder() >= nbPoints) {
                    throw new NumberIsTooLargeException(t.getOrder(), nbPoints, false);
                }

                // compute sample position, trying to be centered if possible
                final double t0 = FastMath.max(FastMath.min(t.getValue(), tMax), tMin) - halfSampleSpan;

                // compute sample points
                final double[] y = new double[nbPoints];
                for (int i = 0; i < nbPoints; ++i) {
                    y[i] = function.value(t0 + i * stepSize);
                }

                // evaluate derivatives
                return evaluate(t, t0, y);

            }

        };
    }

    
    public UnivariateDifferentiableVectorFunction differentiate(final UnivariateVectorFunction function) {
        return new UnivariateDifferentiableVectorFunction() {

            
            public double[]value(final double x) throws MathIllegalArgumentException {
                return function.value(x);
            }

            
            public DerivativeStructure[] value(final DerivativeStructure t)
                throws MathIllegalArgumentException {

                // check we can achieve the requested derivation order with the sample
                if (t.getOrder() >= nbPoints) {
                    throw new NumberIsTooLargeException(t.getOrder(), nbPoints, false);
                }

                // compute sample position, trying to be centered if possible
                final double t0 = FastMath.max(FastMath.min(t.getValue(), tMax), tMin) - halfSampleSpan;

                // compute sample points
                double[][] y = null;
                for (int i = 0; i < nbPoints; ++i) {
                    final double[] v = function.value(t0 + i * stepSize);
                    if (i == 0) {
                        y = new double[v.length][nbPoints];
                    }
                    for (int j = 0; j < v.length; ++j) {
                        y[j][i] = v[j];
                    }
                }

                // evaluate derivatives
                final DerivativeStructure[] value = new DerivativeStructure[y.length];
                for (int j = 0; j < value.length; ++j) {
                    value[j] = evaluate(t, t0, y[j]);
                }

                return value;

            }

        };
    }

    
    public UnivariateDifferentiableMatrixFunction differentiate(final UnivariateMatrixFunction function) {
        return new UnivariateDifferentiableMatrixFunction() {

            
            public double[][]  value(final double x) throws MathIllegalArgumentException {
                return function.value(x);
            }

            
            public DerivativeStructure[][]  value(final DerivativeStructure t)
                throws MathIllegalArgumentException {

                // check we can achieve the requested derivation order with the sample
                if (t.getOrder() >= nbPoints) {
                    throw new NumberIsTooLargeException(t.getOrder(), nbPoints, false);
                }

                // compute sample position, trying to be centered if possible
                final double t0 = FastMath.max(FastMath.min(t.getValue(), tMax), tMin) - halfSampleSpan;

                // compute sample points
                double[][][] y = null;
                for (int i = 0; i < nbPoints; ++i) {
                    final double[][] v = function.value(t0 + i * stepSize);
                    if (i == 0) {
                        y = new double[v.length][v[0].length][nbPoints];
                    }
                    for (int j = 0; j < v.length; ++j) {
                        for (int k = 0; k < v[j].length; ++k) {
                            y[j][k][i] = v[j][k];
                        }
                    }
                }

                // evaluate derivatives
                final DerivativeStructure[][] value = new DerivativeStructure[y.length][y[0].length];
                for (int j = 0; j < value.length; ++j) {
                    for (int k = 0; k < y[j].length; ++k) {
                        value[j][k] = evaluate(t, t0, y[j][k]);
                    }
                }

                return value;

            }

        };
    }

}
