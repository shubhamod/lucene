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
package org.apache.lucene.util.hnsw.math.analysis.interpolation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.UnivariateDifferentiableVectorFunction;
import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialFunction;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.CombinatoricsUtils;


public class HermiteInterpolator implements UnivariateDifferentiableVectorFunction {

    
    private final List<Double> abscissae;

    
    private final List<double[]> topDiagonal;

    
    private final List<double[]> bottomDiagonal;

    
    public HermiteInterpolator() {
        this.abscissae      = new ArrayList<Double>();
        this.topDiagonal    = new ArrayList<double[]>();
        this.bottomDiagonal = new ArrayList<double[]>();
    }

    
    public void addSamplePoint(final double x, final double[] ... value)
        throws ZeroException, MathArithmeticException {

        for (int i = 0; i < value.length; ++i) {

            final double[] y = value[i].clone();
            if (i > 1) {
                double inv = 1.0 / CombinatoricsUtils.factorial(i);
                for (int j = 0; j < y.length; ++j) {
                    y[j] *= inv;
                }
            }

            // update the bottom diagonal of the divided differences array
            final int n = abscissae.size();
            bottomDiagonal.add(n - i, y);
            double[] bottom0 = y;
            for (int j = i; j < n; ++j) {
                final double[] bottom1 = bottomDiagonal.get(n - (j + 1));
                final double inv = 1.0 / (x - abscissae.get(n - (j + 1)));
                if (Double.isInfinite(inv)) {
                    throw new ZeroException(LocalizedFormats.DUPLICATED_ABSCISSA_DIVISION_BY_ZERO, x);
                }
                for (int k = 0; k < y.length; ++k) {
                    bottom1[k] = inv * (bottom0[k] - bottom1[k]);
                }
                bottom0 = bottom1;
            }

            // update the top diagonal of the divided differences array
            topDiagonal.add(bottom0.clone());

            // update the abscissae array
            abscissae.add(x);

        }

    }

    
    public PolynomialFunction[] getPolynomials()
        throws NoDataException {

        // safety check
        checkInterpolation();

        // iteration initialization
        final PolynomialFunction zero = polynomial(0);
        PolynomialFunction[] polynomials = new PolynomialFunction[topDiagonal.get(0).length];
        for (int i = 0; i < polynomials.length; ++i) {
            polynomials[i] = zero;
        }
        PolynomialFunction coeff = polynomial(1);

        // build the polynomials by iterating on the top diagonal of the divided differences array
        for (int i = 0; i < topDiagonal.size(); ++i) {
            double[] tdi = topDiagonal.get(i);
            for (int k = 0; k < polynomials.length; ++k) {
                polynomials[k] = polynomials[k].add(coeff.multiply(polynomial(tdi[k])));
            }
            coeff = coeff.multiply(polynomial(-abscissae.get(i), 1.0));
        }

        return polynomials;

    }

    
    public double[] value(double x)
        throws NoDataException {

        // safety check
        checkInterpolation();

        final double[] value = new double[topDiagonal.get(0).length];
        double valueCoeff = 1;
        for (int i = 0; i < topDiagonal.size(); ++i) {
            double[] dividedDifference = topDiagonal.get(i);
            for (int k = 0; k < value.length; ++k) {
                value[k] += dividedDifference[k] * valueCoeff;
            }
            final double deltaX = x - abscissae.get(i);
            valueCoeff *= deltaX;
        }

        return value;

    }

    
    public DerivativeStructure[] value(final DerivativeStructure x)
        throws NoDataException {

        // safety check
        checkInterpolation();

        final DerivativeStructure[] value = new DerivativeStructure[topDiagonal.get(0).length];
        Arrays.fill(value, x.getField().getZero());
        DerivativeStructure valueCoeff = x.getField().getOne();
        for (int i = 0; i < topDiagonal.size(); ++i) {
            double[] dividedDifference = topDiagonal.get(i);
            for (int k = 0; k < value.length; ++k) {
                value[k] = value[k].add(valueCoeff.multiply(dividedDifference[k]));
            }
            final DerivativeStructure deltaX = x.subtract(abscissae.get(i));
            valueCoeff = valueCoeff.multiply(deltaX);
        }

        return value;

    }

    
    private void checkInterpolation() throws NoDataException {
        if (abscissae.isEmpty()) {
            throw new NoDataException(LocalizedFormats.EMPTY_INTERPOLATION_SAMPLE);
        }
    }

    
    private PolynomialFunction polynomial(double ... c) {
        return new PolynomialFunction(c);
    }

}
