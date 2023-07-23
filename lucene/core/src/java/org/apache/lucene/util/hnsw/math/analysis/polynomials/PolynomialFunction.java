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
package org.apache.lucene.util.hnsw.math.analysis.polynomials;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.analysis.DifferentiableUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.ParametricUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class PolynomialFunction implements UnivariateDifferentiableFunction, DifferentiableUnivariateFunction, Serializable {
    
    private static final long serialVersionUID = -7726511984200295583L;
    
    private final double coefficients[];

    
    public PolynomialFunction(double c[])
        throws NullArgumentException, NoDataException {
        super();
        MathUtils.checkNotNull(c);
        int n = c.length;
        if (n == 0) {
            throw new NoDataException(LocalizedFormats.EMPTY_POLYNOMIALS_COEFFICIENTS_ARRAY);
        }
        while ((n > 1) && (c[n - 1] == 0)) {
            --n;
        }
        this.coefficients = new double[n];
        System.arraycopy(c, 0, this.coefficients, 0, n);
    }

    
    public double value(double x) {
       return evaluate(coefficients, x);
    }

    
    public int degree() {
        return coefficients.length - 1;
    }

    
    public double[] getCoefficients() {
        return coefficients.clone();
    }

    
    protected static double evaluate(double[] coefficients, double argument)
        throws NullArgumentException, NoDataException {
        MathUtils.checkNotNull(coefficients);
        int n = coefficients.length;
        if (n == 0) {
            throw new NoDataException(LocalizedFormats.EMPTY_POLYNOMIALS_COEFFICIENTS_ARRAY);
        }
        double result = coefficients[n - 1];
        for (int j = n - 2; j >= 0; j--) {
            result = argument * result + coefficients[j];
        }
        return result;
    }


    
    public DerivativeStructure value(final DerivativeStructure t)
        throws NullArgumentException, NoDataException {
        MathUtils.checkNotNull(coefficients);
        int n = coefficients.length;
        if (n == 0) {
            throw new NoDataException(LocalizedFormats.EMPTY_POLYNOMIALS_COEFFICIENTS_ARRAY);
        }
        DerivativeStructure result =
                new DerivativeStructure(t.getFreeParameters(), t.getOrder(), coefficients[n - 1]);
        for (int j = n - 2; j >= 0; j--) {
            result = result.multiply(t).add(coefficients[j]);
        }
        return result;
    }

    
    public PolynomialFunction add(final PolynomialFunction p) {
        // identify the lowest degree polynomial
        final int lowLength  = FastMath.min(coefficients.length, p.coefficients.length);
        final int highLength = FastMath.max(coefficients.length, p.coefficients.length);

        // build the coefficients array
        double[] newCoefficients = new double[highLength];
        for (int i = 0; i < lowLength; ++i) {
            newCoefficients[i] = coefficients[i] + p.coefficients[i];
        }
        System.arraycopy((coefficients.length < p.coefficients.length) ?
                         p.coefficients : coefficients,
                         lowLength,
                         newCoefficients, lowLength,
                         highLength - lowLength);

        return new PolynomialFunction(newCoefficients);
    }

    
    public PolynomialFunction subtract(final PolynomialFunction p) {
        // identify the lowest degree polynomial
        int lowLength  = FastMath.min(coefficients.length, p.coefficients.length);
        int highLength = FastMath.max(coefficients.length, p.coefficients.length);

        // build the coefficients array
        double[] newCoefficients = new double[highLength];
        for (int i = 0; i < lowLength; ++i) {
            newCoefficients[i] = coefficients[i] - p.coefficients[i];
        }
        if (coefficients.length < p.coefficients.length) {
            for (int i = lowLength; i < highLength; ++i) {
                newCoefficients[i] = -p.coefficients[i];
            }
        } else {
            System.arraycopy(coefficients, lowLength, newCoefficients, lowLength,
                             highLength - lowLength);
        }

        return new PolynomialFunction(newCoefficients);
    }

    
    public PolynomialFunction negate() {
        double[] newCoefficients = new double[coefficients.length];
        for (int i = 0; i < coefficients.length; ++i) {
            newCoefficients[i] = -coefficients[i];
        }
        return new PolynomialFunction(newCoefficients);
    }

    
    public PolynomialFunction multiply(final PolynomialFunction p) {
        double[] newCoefficients = new double[coefficients.length + p.coefficients.length - 1];

        for (int i = 0; i < newCoefficients.length; ++i) {
            newCoefficients[i] = 0.0;
            for (int j = FastMath.max(0, i + 1 - p.coefficients.length);
                 j < FastMath.min(coefficients.length, i + 1);
                 ++j) {
                newCoefficients[i] += coefficients[j] * p.coefficients[i-j];
            }
        }

        return new PolynomialFunction(newCoefficients);
    }

    
    protected static double[] differentiate(double[] coefficients)
        throws NullArgumentException, NoDataException {
        MathUtils.checkNotNull(coefficients);
        int n = coefficients.length;
        if (n == 0) {
            throw new NoDataException(LocalizedFormats.EMPTY_POLYNOMIALS_COEFFICIENTS_ARRAY);
        }
        if (n == 1) {
            return new double[]{0};
        }
        double[] result = new double[n - 1];
        for (int i = n - 1; i > 0; i--) {
            result[i - 1] = i * coefficients[i];
        }
        return result;
    }

    
    public PolynomialFunction polynomialDerivative() {
        return new PolynomialFunction(differentiate(coefficients));
    }

    
    public UnivariateFunction derivative() {
        return polynomialDerivative();
    }

    
    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        if (coefficients[0] == 0.0) {
            if (coefficients.length == 1) {
                return "0";
            }
        } else {
            s.append(toString(coefficients[0]));
        }

        for (int i = 1; i < coefficients.length; ++i) {
            if (coefficients[i] != 0) {
                if (s.length() > 0) {
                    if (coefficients[i] < 0) {
                        s.append(" - ");
                    } else {
                        s.append(" + ");
                    }
                } else {
                    if (coefficients[i] < 0) {
                        s.append("-");
                    }
                }

                double absAi = FastMath.abs(coefficients[i]);
                if ((absAi - 1) != 0) {
                    s.append(toString(absAi));
                    s.append(' ');
                }

                s.append("x");
                if (i > 1) {
                    s.append('^');
                    s.append(Integer.toString(i));
                }
            }
        }

        return s.toString();
    }

    
    private static String toString(double coeff) {
        final String c = Double.toString(coeff);
        if (c.endsWith(".0")) {
            return c.substring(0, c.length() - 2);
        } else {
            return c;
        }
    }

    
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + Arrays.hashCode(coefficients);
        return result;
    }

    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof PolynomialFunction)) {
            return false;
        }
        PolynomialFunction other = (PolynomialFunction) obj;
        if (!Arrays.equals(coefficients, other.coefficients)) {
            return false;
        }
        return true;
    }

    
    public static class Parametric implements ParametricUnivariateFunction {
        
        public double[] gradient(double x, double ... parameters) {
            final double[] gradient = new double[parameters.length];
            double xn = 1.0;
            for (int i = 0; i < parameters.length; ++i) {
                gradient[i] = xn;
                xn *= x;
            }
            return gradient;
        }

        
        public double value(final double x, final double ... parameters)
            throws NoDataException {
            return PolynomialFunction.evaluate(parameters, x);
        }
    }
}
