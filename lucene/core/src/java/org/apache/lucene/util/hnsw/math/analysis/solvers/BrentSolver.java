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


import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class BrentSolver extends AbstractUnivariateSolver {

    
    private static final double DEFAULT_ABSOLUTE_ACCURACY = 1e-6;

    
    public BrentSolver() {
        this(DEFAULT_ABSOLUTE_ACCURACY);
    }
    
    public BrentSolver(double absoluteAccuracy) {
        super(absoluteAccuracy);
    }
    
    public BrentSolver(double relativeAccuracy,
                       double absoluteAccuracy) {
        super(relativeAccuracy, absoluteAccuracy);
    }
    
    public BrentSolver(double relativeAccuracy,
                       double absoluteAccuracy,
                       double functionValueAccuracy) {
        super(relativeAccuracy, absoluteAccuracy, functionValueAccuracy);
    }

    
    @Override
    protected double doSolve()
        throws NoBracketingException,
               TooManyEvaluationsException,
               NumberIsTooLargeException {
        double min = getMin();
        double max = getMax();
        final double initial = getStartValue();
        final double functionValueAccuracy = getFunctionValueAccuracy();

        verifySequence(min, initial, max);

        // Return the initial guess if it is good enough.
        double yInitial = computeObjectiveValue(initial);
        if (FastMath.abs(yInitial) <= functionValueAccuracy) {
            return initial;
        }

        // Return the first endpoint if it is good enough.
        double yMin = computeObjectiveValue(min);
        if (FastMath.abs(yMin) <= functionValueAccuracy) {
            return min;
        }

        // Reduce interval if min and initial bracket the root.
        if (yInitial * yMin < 0) {
            return brent(min, initial, yMin, yInitial);
        }

        // Return the second endpoint if it is good enough.
        double yMax = computeObjectiveValue(max);
        if (FastMath.abs(yMax) <= functionValueAccuracy) {
            return max;
        }

        // Reduce interval if initial and max bracket the root.
        if (yInitial * yMax < 0) {
            return brent(initial, max, yInitial, yMax);
        }

        throw new NoBracketingException(min, max, yMin, yMax);
    }

    
    private double brent(double lo, double hi,
                         double fLo, double fHi) {
        double a = lo;
        double fa = fLo;
        double b = hi;
        double fb = fHi;
        double c = a;
        double fc = fa;
        double d = b - a;
        double e = d;

        final double t = getAbsoluteAccuracy();
        final double eps = getRelativeAccuracy();

        while (true) {
            if (FastMath.abs(fc) < FastMath.abs(fb)) {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }

            final double tol = 2 * eps * FastMath.abs(b) + t;
            final double m = 0.5 * (c - b);

            if (FastMath.abs(m) <= tol ||
                Precision.equals(fb, 0))  {
                return b;
            }
            if (FastMath.abs(e) < tol ||
                FastMath.abs(fa) <= FastMath.abs(fb)) {
                // Force bisection.
                d = m;
                e = d;
            } else {
                double s = fb / fa;
                double p;
                double q;
                // The equality test (a == c) is intentional,
                // it is part of the original Brent's method and
                // it should NOT be replaced by proximity test.
                if (a == c) {
                    // Linear interpolation.
                    p = 2 * m * s;
                    q = 1 - s;
                } else {
                    // Inverse quadratic interpolation.
                    q = fa / fc;
                    final double r = fb / fc;
                    p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
                    q = (q - 1) * (r - 1) * (s - 1);
                }
                if (p > 0) {
                    q = -q;
                } else {
                    p = -p;
                }
                s = e;
                e = d;
                if (p >= 1.5 * m * q - FastMath.abs(tol * q) ||
                    p >= FastMath.abs(0.5 * s * q)) {
                    // Inverse quadratic interpolation gives a value
                    // in the wrong direction, or progress is slow.
                    // Fall back to bisection.
                    d = m;
                    e = d;
                } else {
                    d = p / q;
                }
            }
            a = b;
            fa = fb;

            if (FastMath.abs(d) > tol) {
                b += d;
            } else if (m > 0) {
                b += tol;
            } else {
                b -= tol;
            }
            fb = computeObjectiveValue(b);
            if ((fb > 0 && fc > 0) ||
                (fb <= 0 && fc <= 0)) {
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }
        }
    }
}
