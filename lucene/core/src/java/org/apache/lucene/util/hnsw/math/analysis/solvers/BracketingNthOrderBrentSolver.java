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
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class BracketingNthOrderBrentSolver
    extends AbstractUnivariateSolver
    implements BracketedUnivariateSolver<UnivariateFunction> {

    
    private static final double DEFAULT_ABSOLUTE_ACCURACY = 1e-6;

    
    private static final int DEFAULT_MAXIMAL_ORDER = 5;

    
    private static final int MAXIMAL_AGING = 2;

    
    private static final double REDUCTION_FACTOR = 1.0 / 16.0;

    
    private final int maximalOrder;

    
    private AllowedSolution allowed;

    
    public BracketingNthOrderBrentSolver() {
        this(DEFAULT_ABSOLUTE_ACCURACY, DEFAULT_MAXIMAL_ORDER);
    }

    
    public BracketingNthOrderBrentSolver(final double absoluteAccuracy,
                                         final int maximalOrder)
        throws NumberIsTooSmallException {
        super(absoluteAccuracy);
        if (maximalOrder < 2) {
            throw new NumberIsTooSmallException(maximalOrder, 2, true);
        }
        this.maximalOrder = maximalOrder;
        this.allowed = AllowedSolution.ANY_SIDE;
    }

    
    public BracketingNthOrderBrentSolver(final double relativeAccuracy,
                                         final double absoluteAccuracy,
                                         final int maximalOrder)
        throws NumberIsTooSmallException {
        super(relativeAccuracy, absoluteAccuracy);
        if (maximalOrder < 2) {
            throw new NumberIsTooSmallException(maximalOrder, 2, true);
        }
        this.maximalOrder = maximalOrder;
        this.allowed = AllowedSolution.ANY_SIDE;
    }

    
    public BracketingNthOrderBrentSolver(final double relativeAccuracy,
                                         final double absoluteAccuracy,
                                         final double functionValueAccuracy,
                                         final int maximalOrder)
        throws NumberIsTooSmallException {
        super(relativeAccuracy, absoluteAccuracy, functionValueAccuracy);
        if (maximalOrder < 2) {
            throw new NumberIsTooSmallException(maximalOrder, 2, true);
        }
        this.maximalOrder = maximalOrder;
        this.allowed = AllowedSolution.ANY_SIDE;
    }

    
    public int getMaximalOrder() {
        return maximalOrder;
    }

    
    @Override
    protected double doSolve()
        throws TooManyEvaluationsException,
               NumberIsTooLargeException,
               NoBracketingException {
        // prepare arrays with the first points
        final double[] x = new double[maximalOrder + 1];
        final double[] y = new double[maximalOrder + 1];
        x[0] = getMin();
        x[1] = getStartValue();
        x[2] = getMax();
        verifySequence(x[0], x[1], x[2]);

        // evaluate initial guess
        y[1] = computeObjectiveValue(x[1]);
        if (Precision.equals(y[1], 0.0, 1)) {
            // return the initial guess if it is a perfect root.
            return x[1];
        }

        // evaluate first  endpoint
        y[0] = computeObjectiveValue(x[0]);
        if (Precision.equals(y[0], 0.0, 1)) {
            // return the first endpoint if it is a perfect root.
            return x[0];
        }

        int nbPoints;
        int signChangeIndex;
        if (y[0] * y[1] < 0) {

            // reduce interval if it brackets the root
            nbPoints        = 2;
            signChangeIndex = 1;

        } else {

            // evaluate second endpoint
            y[2] = computeObjectiveValue(x[2]);
            if (Precision.equals(y[2], 0.0, 1)) {
                // return the second endpoint if it is a perfect root.
                return x[2];
            }

            if (y[1] * y[2] < 0) {
                // use all computed point as a start sampling array for solving
                nbPoints        = 3;
                signChangeIndex = 2;
            } else {
                throw new NoBracketingException(x[0], x[2], y[0], y[2]);
            }

        }

        // prepare a work array for inverse polynomial interpolation
        final double[] tmpX = new double[x.length];

        // current tightest bracketing of the root
        double xA    = x[signChangeIndex - 1];
        double yA    = y[signChangeIndex - 1];
        double absYA = FastMath.abs(yA);
        int agingA   = 0;
        double xB    = x[signChangeIndex];
        double yB    = y[signChangeIndex];
        double absYB = FastMath.abs(yB);
        int agingB   = 0;

        // search loop
        while (true) {

            // check convergence of bracketing interval
            final double xTol = getAbsoluteAccuracy() +
                                getRelativeAccuracy() * FastMath.max(FastMath.abs(xA), FastMath.abs(xB));
            if (((xB - xA) <= xTol) || (FastMath.max(absYA, absYB) < getFunctionValueAccuracy())) {
                switch (allowed) {
                case ANY_SIDE :
                    return absYA < absYB ? xA : xB;
                case LEFT_SIDE :
                    return xA;
                case RIGHT_SIDE :
                    return xB;
                case BELOW_SIDE :
                    return (yA <= 0) ? xA : xB;
                case ABOVE_SIDE :
                    return (yA <  0) ? xB : xA;
                default :
                    // this should never happen
                    throw new MathInternalError();
                }
            }

            // target for the next evaluation point
            double targetY;
            if (agingA >= MAXIMAL_AGING) {
                // we keep updating the high bracket, try to compensate this
                final int p = agingA - MAXIMAL_AGING;
                final double weightA = (1 << p) - 1;
                final double weightB = p + 1;
                targetY = (weightA * yA - weightB * REDUCTION_FACTOR * yB) / (weightA + weightB);
            } else if (agingB >= MAXIMAL_AGING) {
                // we keep updating the low bracket, try to compensate this
                final int p = agingB - MAXIMAL_AGING;
                final double weightA = p + 1;
                final double weightB = (1 << p) - 1;
                targetY = (weightB * yB - weightA * REDUCTION_FACTOR * yA) / (weightA + weightB);
            } else {
                // bracketing is balanced, try to find the root itself
                targetY = 0;
            }

            // make a few attempts to guess a root,
            double nextX;
            int start = 0;
            int end   = nbPoints;
            do {

                // guess a value for current target, using inverse polynomial interpolation
                System.arraycopy(x, start, tmpX, start, end - start);
                nextX = guessX(targetY, tmpX, y, start, end);

                if (!((nextX > xA) && (nextX < xB))) {
                    // the guessed root is not strictly inside of the tightest bracketing interval

                    // the guessed root is either not strictly inside the interval or it
                    // is a NaN (which occurs when some sampling points share the same y)
                    // we try again with a lower interpolation order
                    if (signChangeIndex - start >= end - signChangeIndex) {
                        // we have more points before the sign change, drop the lowest point
                        ++start;
                    } else {
                        // we have more points after sign change, drop the highest point
                        --end;
                    }

                    // we need to do one more attempt
                    nextX = Double.NaN;

                }

            } while (Double.isNaN(nextX) && (end - start > 1));

            if (Double.isNaN(nextX)) {
                // fall back to bisection
                nextX = xA + 0.5 * (xB - xA);
                start = signChangeIndex - 1;
                end   = signChangeIndex;
            }

            // evaluate the function at the guessed root
            final double nextY = computeObjectiveValue(nextX);
            if (Precision.equals(nextY, 0.0, 1)) {
                // we have found an exact root, since it is not an approximation
                // we don't need to bother about the allowed solutions setting
                return nextX;
            }

            if ((nbPoints > 2) && (end - start != nbPoints)) {

                // we have been forced to ignore some points to keep bracketing,
                // they are probably too far from the root, drop them from now on
                nbPoints = end - start;
                System.arraycopy(x, start, x, 0, nbPoints);
                System.arraycopy(y, start, y, 0, nbPoints);
                signChangeIndex -= start;

            } else  if (nbPoints == x.length) {

                // we have to drop one point in order to insert the new one
                nbPoints--;

                // keep the tightest bracketing interval as centered as possible
                if (signChangeIndex >= (x.length + 1) / 2) {
                    // we drop the lowest point, we have to shift the arrays and the index
                    System.arraycopy(x, 1, x, 0, nbPoints);
                    System.arraycopy(y, 1, y, 0, nbPoints);
                    --signChangeIndex;
                }

            }

            // insert the last computed point
            //(by construction, we know it lies inside the tightest bracketing interval)
            System.arraycopy(x, signChangeIndex, x, signChangeIndex + 1, nbPoints - signChangeIndex);
            x[signChangeIndex] = nextX;
            System.arraycopy(y, signChangeIndex, y, signChangeIndex + 1, nbPoints - signChangeIndex);
            y[signChangeIndex] = nextY;
            ++nbPoints;

            // update the bracketing interval
            if (nextY * yA <= 0) {
                // the sign change occurs before the inserted point
                xB = nextX;
                yB = nextY;
                absYB = FastMath.abs(yB);
                ++agingA;
                agingB = 0;
            } else {
                // the sign change occurs after the inserted point
                xA = nextX;
                yA = nextY;
                absYA = FastMath.abs(yA);
                agingA = 0;
                ++agingB;

                // update the sign change index
                signChangeIndex++;

            }

        }

    }

    
    private double guessX(final double targetY, final double[] x, final double[] y,
                          final int start, final int end) {

        // compute Q Newton coefficients by divided differences
        for (int i = start; i < end - 1; ++i) {
            final int delta = i + 1 - start;
            for (int j = end - 1; j > i; --j) {
                x[j] = (x[j] - x[j-1]) / (y[j] - y[j - delta]);
            }
        }

        // evaluate Q(targetY)
        double x0 = 0;
        for (int j = end - 1; j >= start; --j) {
            x0 = x[j] + x0 * (targetY - y[j]);
        }

        return x0;

    }

    
    public double solve(int maxEval, UnivariateFunction f, double min,
                        double max, AllowedSolution allowedSolution)
        throws TooManyEvaluationsException,
               NumberIsTooLargeException,
               NoBracketingException {
        this.allowed = allowedSolution;
        return super.solve(maxEval, f, min, max);
    }

    
    public double solve(int maxEval, UnivariateFunction f, double min,
                        double max, double startValue,
                        AllowedSolution allowedSolution)
        throws TooManyEvaluationsException,
               NumberIsTooLargeException,
               NoBracketingException {
        this.allowed = allowedSolution;
        return super.solve(maxEval, f, min, max, startValue);
    }

}
