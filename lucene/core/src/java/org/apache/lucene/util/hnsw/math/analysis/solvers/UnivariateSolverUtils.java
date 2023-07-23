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

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class UnivariateSolverUtils {
    
    private UnivariateSolverUtils() {}

    
    public static double solve(UnivariateFunction function, double x0, double x1)
        throws NullArgumentException,
               NoBracketingException {
        if (function == null) {
            throw new NullArgumentException(LocalizedFormats.FUNCTION);
        }
        final UnivariateSolver solver = new BrentSolver();
        return solver.solve(Integer.MAX_VALUE, function, x0, x1);
    }

    
    public static double solve(UnivariateFunction function,
                               double x0, double x1,
                               double absoluteAccuracy)
        throws NullArgumentException,
               NoBracketingException {
        if (function == null) {
            throw new NullArgumentException(LocalizedFormats.FUNCTION);
        }
        final UnivariateSolver solver = new BrentSolver(absoluteAccuracy);
        return solver.solve(Integer.MAX_VALUE, function, x0, x1);
    }

    
    public static double forceSide(final int maxEval, final UnivariateFunction f,
                                   final BracketedUnivariateSolver<UnivariateFunction> bracketing,
                                   final double baseRoot, final double min, final double max,
                                   final AllowedSolution allowedSolution)
        throws NoBracketingException {

        if (allowedSolution == AllowedSolution.ANY_SIDE) {
            // no further bracketing required
            return baseRoot;
        }

        // find a very small interval bracketing the root
        final double step = FastMath.max(bracketing.getAbsoluteAccuracy(),
                                         FastMath.abs(baseRoot * bracketing.getRelativeAccuracy()));
        double xLo        = FastMath.max(min, baseRoot - step);
        double fLo        = f.value(xLo);
        double xHi        = FastMath.min(max, baseRoot + step);
        double fHi        = f.value(xHi);
        int remainingEval = maxEval - 2;
        while (remainingEval > 0) {

            if ((fLo >= 0 && fHi <= 0) || (fLo <= 0 && fHi >= 0)) {
                // compute the root on the selected side
                return bracketing.solve(remainingEval, f, xLo, xHi, baseRoot, allowedSolution);
            }

            // try increasing the interval
            boolean changeLo = false;
            boolean changeHi = false;
            if (fLo < fHi) {
                // increasing function
                if (fLo >= 0) {
                    changeLo = true;
                } else {
                    changeHi = true;
                }
            } else if (fLo > fHi) {
                // decreasing function
                if (fLo <= 0) {
                    changeLo = true;
                } else {
                    changeHi = true;
                }
            } else {
                // unknown variation
                changeLo = true;
                changeHi = true;
            }

            // update the lower bound
            if (changeLo) {
                xLo = FastMath.max(min, xLo - step);
                fLo  = f.value(xLo);
                remainingEval--;
            }

            // update the higher bound
            if (changeHi) {
                xHi = FastMath.min(max, xHi + step);
                fHi  = f.value(xHi);
                remainingEval--;
            }

        }

        throw new NoBracketingException(LocalizedFormats.FAILED_BRACKETING,
                                        xLo, xHi, fLo, fHi,
                                        maxEval - remainingEval, maxEval, baseRoot,
                                        min, max);

    }

    
    public static double[] bracket(UnivariateFunction function,
                                   double initial,
                                   double lowerBound, double upperBound)
        throws NullArgumentException,
               NotStrictlyPositiveException,
               NoBracketingException {
        return bracket(function, initial, lowerBound, upperBound, 1.0, 1.0, Integer.MAX_VALUE);
    }

     
    public static double[] bracket(UnivariateFunction function,
                                   double initial,
                                   double lowerBound, double upperBound,
                                   int maximumIterations)
        throws NullArgumentException,
               NotStrictlyPositiveException,
               NoBracketingException {
        return bracket(function, initial, lowerBound, upperBound, 1.0, 1.0, maximumIterations);
    }

    
    public static double[] bracket(final UnivariateFunction function, final double initial,
                                   final double lowerBound, final double upperBound,
                                   final double q, final double r, final int maximumIterations)
        throws NoBracketingException {

        if (function == null) {
            throw new NullArgumentException(LocalizedFormats.FUNCTION);
        }
        if (q <= 0)  {
            throw new NotStrictlyPositiveException(q);
        }
        if (maximumIterations <= 0)  {
            throw new NotStrictlyPositiveException(LocalizedFormats.INVALID_MAX_ITERATIONS, maximumIterations);
        }
        verifySequence(lowerBound, initial, upperBound);

        // initialize the recurrence
        double a     = initial;
        double b     = initial;
        double fa    = Double.NaN;
        double fb    = Double.NaN;
        double delta = 0;

        for (int numIterations = 0;
             (numIterations < maximumIterations) && (a > lowerBound || b < upperBound);
             ++numIterations) {

            final double previousA  = a;
            final double previousFa = fa;
            final double previousB  = b;
            final double previousFb = fb;

            delta = r * delta + q;
            a     = FastMath.max(initial - delta, lowerBound);
            b     = FastMath.min(initial + delta, upperBound);
            fa    = function.value(a);
            fb    = function.value(b);

            if (numIterations == 0) {
                // at first iteration, we don't have a previous interval
                // we simply compare both sides of the initial interval
                if (fa * fb <= 0) {
                    // the first interval already brackets a root
                    return new double[] { a, b };
                }
            } else {
                // we have a previous interval with constant sign and expand it,
                // we expect sign changes to occur at boundaries
                if (fa * previousFa <= 0) {
                    // sign change detected at near lower bound
                    return new double[] { a, previousA };
                } else if (fb * previousFb <= 0) {
                    // sign change detected at near upper bound
                    return new double[] { previousB, b };
                }
            }

        }

        // no bracketing found
        throw new NoBracketingException(a, b, fa, fb);

    }

    
    public static double midpoint(double a, double b) {
        return (a + b) * 0.5;
    }

    
    public static boolean isBracketing(UnivariateFunction function,
                                       final double lower,
                                       final double upper)
        throws NullArgumentException {
        if (function == null) {
            throw new NullArgumentException(LocalizedFormats.FUNCTION);
        }
        final double fLo = function.value(lower);
        final double fHi = function.value(upper);
        return (fLo >= 0 && fHi <= 0) || (fLo <= 0 && fHi >= 0);
    }

    
    public static boolean isSequence(final double start,
                                     final double mid,
                                     final double end) {
        return (start < mid) && (mid < end);
    }

    
    public static void verifyInterval(final double lower,
                                      final double upper)
        throws NumberIsTooLargeException {
        if (lower >= upper) {
            throw new NumberIsTooLargeException(LocalizedFormats.ENDPOINTS_NOT_AN_INTERVAL,
                                                lower, upper, false);
        }
    }

    
    public static void verifySequence(final double lower,
                                      final double initial,
                                      final double upper)
        throws NumberIsTooLargeException {
        verifyInterval(lower, initial);
        verifyInterval(initial, upper);
    }

    
    public static void verifyBracketing(UnivariateFunction function,
                                        final double lower,
                                        final double upper)
        throws NullArgumentException,
               NoBracketingException {
        if (function == null) {
            throw new NullArgumentException(LocalizedFormats.FUNCTION);
        }
        verifyInterval(lower, upper);
        if (!isBracketing(function, lower, upper)) {
            throw new NoBracketingException(lower, upper,
                                            function.value(lower),
                                            function.value(upper));
        }
    }
}
