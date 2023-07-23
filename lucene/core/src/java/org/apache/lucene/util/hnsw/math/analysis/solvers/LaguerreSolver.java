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
package org.apache.lucene.util.hnsw.math.analysis.solvers;

import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialFunction;
import org.apache.lucene.util.hnsw.math.complex.Complex;
import org.apache.lucene.util.hnsw.math.complex.ComplexUtils;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class LaguerreSolver extends AbstractPolynomialSolver {
    
    private static final double DEFAULT_ABSOLUTE_ACCURACY = 1e-6;
    
    private final ComplexSolver complexSolver = new ComplexSolver();

    
    public LaguerreSolver() {
        this(DEFAULT_ABSOLUTE_ACCURACY);
    }
    
    public LaguerreSolver(double absoluteAccuracy) {
        super(absoluteAccuracy);
    }
    
    public LaguerreSolver(double relativeAccuracy,
                          double absoluteAccuracy) {
        super(relativeAccuracy, absoluteAccuracy);
    }
    
    public LaguerreSolver(double relativeAccuracy,
                          double absoluteAccuracy,
                          double functionValueAccuracy) {
        super(relativeAccuracy, absoluteAccuracy, functionValueAccuracy);
    }

    
    @Override
    public double doSolve()
        throws TooManyEvaluationsException,
               NumberIsTooLargeException,
               NoBracketingException {
        final double min = getMin();
        final double max = getMax();
        final double initial = getStartValue();
        final double functionValueAccuracy = getFunctionValueAccuracy();

        verifySequence(min, initial, max);

        // Return the initial guess if it is good enough.
        final double yInitial = computeObjectiveValue(initial);
        if (FastMath.abs(yInitial) <= functionValueAccuracy) {
            return initial;
        }

        // Return the first endpoint if it is good enough.
        final double yMin = computeObjectiveValue(min);
        if (FastMath.abs(yMin) <= functionValueAccuracy) {
            return min;
        }

        // Reduce interval if min and initial bracket the root.
        if (yInitial * yMin < 0) {
            return laguerre(min, initial, yMin, yInitial);
        }

        // Return the second endpoint if it is good enough.
        final double yMax = computeObjectiveValue(max);
        if (FastMath.abs(yMax) <= functionValueAccuracy) {
            return max;
        }

        // Reduce interval if initial and max bracket the root.
        if (yInitial * yMax < 0) {
            return laguerre(initial, max, yInitial, yMax);
        }

        throw new NoBracketingException(min, max, yMin, yMax);
    }

    
    @Deprecated
    public double laguerre(double lo, double hi,
                           double fLo, double fHi) {
        final Complex c[] = ComplexUtils.convertToComplex(getCoefficients());

        final Complex initial = new Complex(0.5 * (lo + hi), 0);
        final Complex z = complexSolver.solve(c, initial);
        if (complexSolver.isRoot(lo, hi, z)) {
            return z.getReal();
        } else {
            double r = Double.NaN;
            // Solve all roots and select the one we are seeking.
            Complex[] root = complexSolver.solveAll(c, initial);
            for (int i = 0; i < root.length; i++) {
                if (complexSolver.isRoot(lo, hi, root[i])) {
                    r = root[i].getReal();
                    break;
                }
            }
            return r;
        }
    }

    
    public Complex[] solveAllComplex(double[] coefficients,
                                     double initial)
        throws NullArgumentException,
               NoDataException,
               TooManyEvaluationsException {
       return solveAllComplex(coefficients, initial, Integer.MAX_VALUE);
    }

    
    public Complex[] solveAllComplex(double[] coefficients,
                                     double initial, int maxEval)
        throws NullArgumentException,
               NoDataException,
               TooManyEvaluationsException {
        setup(maxEval,
              new PolynomialFunction(coefficients),
              Double.NEGATIVE_INFINITY,
              Double.POSITIVE_INFINITY,
              initial);
        return complexSolver.solveAll(ComplexUtils.convertToComplex(coefficients),
                                      new Complex(initial, 0d));
    }

    
    public Complex solveComplex(double[] coefficients,
                                double initial)
        throws NullArgumentException,
               NoDataException,
               TooManyEvaluationsException {
       return solveComplex(coefficients, initial, Integer.MAX_VALUE);
    }

    
    public Complex solveComplex(double[] coefficients,
                                double initial, int maxEval)
        throws NullArgumentException,
               NoDataException,
               TooManyEvaluationsException {
        setup(maxEval,
              new PolynomialFunction(coefficients),
              Double.NEGATIVE_INFINITY,
              Double.POSITIVE_INFINITY,
              initial);
        return complexSolver.solve(ComplexUtils.convertToComplex(coefficients),
                                   new Complex(initial, 0d));
    }

    
    private class ComplexSolver {
        
        public boolean isRoot(double min, double max, Complex z) {
            if (isSequence(min, z.getReal(), max)) {
                double tolerance = FastMath.max(getRelativeAccuracy() * z.abs(), getAbsoluteAccuracy());
                return (FastMath.abs(z.getImaginary()) <= tolerance) ||
                     (z.abs() <= getFunctionValueAccuracy());
            }
            return false;
        }

        
        public Complex[] solveAll(Complex coefficients[], Complex initial)
            throws NullArgumentException,
                   NoDataException,
                   TooManyEvaluationsException {
            if (coefficients == null) {
                throw new NullArgumentException();
            }
            final int n = coefficients.length - 1;
            if (n == 0) {
                throw new NoDataException(LocalizedFormats.POLYNOMIAL);
            }
            // Coefficients for deflated polynomial.
            final Complex c[] = new Complex[n + 1];
            for (int i = 0; i <= n; i++) {
                c[i] = coefficients[i];
            }

            // Solve individual roots successively.
            final Complex root[] = new Complex[n];
            for (int i = 0; i < n; i++) {
                final Complex subarray[] = new Complex[n - i + 1];
                System.arraycopy(c, 0, subarray, 0, subarray.length);
                root[i] = solve(subarray, initial);
                // Polynomial deflation using synthetic division.
                Complex newc = c[n - i];
                Complex oldc = null;
                for (int j = n - i - 1; j >= 0; j--) {
                    oldc = c[j];
                    c[j] = newc;
                    newc = oldc.add(newc.multiply(root[i]));
                }
            }

            return root;
        }

        
        public Complex solve(Complex coefficients[], Complex initial)
            throws NullArgumentException,
                   NoDataException,
                   TooManyEvaluationsException {
            if (coefficients == null) {
                throw new NullArgumentException();
            }

            final int n = coefficients.length - 1;
            if (n == 0) {
                throw new NoDataException(LocalizedFormats.POLYNOMIAL);
            }

            final double absoluteAccuracy = getAbsoluteAccuracy();
            final double relativeAccuracy = getRelativeAccuracy();
            final double functionValueAccuracy = getFunctionValueAccuracy();

            final Complex nC  = new Complex(n, 0);
            final Complex n1C = new Complex(n - 1, 0);

            Complex z = initial;
            Complex oldz = new Complex(Double.POSITIVE_INFINITY,
                                       Double.POSITIVE_INFINITY);
            while (true) {
                // Compute pv (polynomial value), dv (derivative value), and
                // d2v (second derivative value) simultaneously.
                Complex pv = coefficients[n];
                Complex dv = Complex.ZERO;
                Complex d2v = Complex.ZERO;
                for (int j = n-1; j >= 0; j--) {
                    d2v = dv.add(z.multiply(d2v));
                    dv = pv.add(z.multiply(dv));
                    pv = coefficients[j].add(z.multiply(pv));
                }
                d2v = d2v.multiply(new Complex(2.0, 0.0));

                // Check for convergence.
                final double tolerance = FastMath.max(relativeAccuracy * z.abs(),
                                                      absoluteAccuracy);
                if ((z.subtract(oldz)).abs() <= tolerance) {
                    return z;
                }
                if (pv.abs() <= functionValueAccuracy) {
                    return z;
                }

                // Now pv != 0, calculate the new approximation.
                final Complex G = dv.divide(pv);
                final Complex G2 = G.multiply(G);
                final Complex H = G2.subtract(d2v.divide(pv));
                final Complex delta = n1C.multiply((nC.multiply(H)).subtract(G2));
                // Choose a denominator larger in magnitude.
                final Complex deltaSqrt = delta.sqrt();
                final Complex dplus = G.add(deltaSqrt);
                final Complex dminus = G.subtract(deltaSqrt);
                final Complex denominator = dplus.abs() > dminus.abs() ? dplus : dminus;
                // Perturb z if denominator is zero, for instance,
                // p(x) = x^3 + 1, z = 0.
                if (denominator.equals(new Complex(0.0, 0.0))) {
                    z = z.add(new Complex(absoluteAccuracy, absoluteAccuracy));
                    oldz = new Complex(Double.POSITIVE_INFINITY,
                                       Double.POSITIVE_INFINITY);
                } else {
                    oldz = z;
                    z = z.subtract(nC.divide(denominator));
                }
                incrementEvaluationCount();
            }
        }
    }
}
