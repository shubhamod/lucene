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


import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.analysis.RealFieldUnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.util.IntegerSequence;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class FieldBracketingNthOrderBrentSolver<T extends RealFieldElement<T>>
    implements BracketedRealFieldUnivariateSolver<T> {

   
    private static final int MAXIMAL_AGING = 2;

    
    private final Field<T> field;

    
    private final int maximalOrder;

    
    private final T functionValueAccuracy;

    
    private final T absoluteAccuracy;

    
    private final T relativeAccuracy;

    
    private IntegerSequence.Incrementor evaluations;

    
    public FieldBracketingNthOrderBrentSolver(final T relativeAccuracy,
                                              final T absoluteAccuracy,
                                              final T functionValueAccuracy,
                                              final int maximalOrder)
        throws NumberIsTooSmallException {
        if (maximalOrder < 2) {
            throw new NumberIsTooSmallException(maximalOrder, 2, true);
        }
        this.field                 = relativeAccuracy.getField();
        this.maximalOrder          = maximalOrder;
        this.absoluteAccuracy      = absoluteAccuracy;
        this.relativeAccuracy      = relativeAccuracy;
        this.functionValueAccuracy = functionValueAccuracy;
        this.evaluations           = IntegerSequence.Incrementor.create();
    }

    
    public int getMaximalOrder() {
        return maximalOrder;
    }

    
    public int getMaxEvaluations() {
        return evaluations.getMaximalCount();
    }

    
    public int getEvaluations() {
        return evaluations.getCount();
    }

    
    public T getAbsoluteAccuracy() {
        return absoluteAccuracy;
    }

    
    public T getRelativeAccuracy() {
        return relativeAccuracy;
    }

    
    public T getFunctionValueAccuracy() {
        return functionValueAccuracy;
    }

    
    public T solve(final int maxEval, final RealFieldUnivariateFunction<T> f,
                   final T min, final T max, final AllowedSolution allowedSolution)
        throws NullArgumentException, NoBracketingException {
        return solve(maxEval, f, min, max, min.add(max).divide(2), allowedSolution);
    }

    
    public T solve(final int maxEval, final RealFieldUnivariateFunction<T> f,
                   final T min, final T max, final T startValue,
                   final AllowedSolution allowedSolution)
        throws NullArgumentException, NoBracketingException {

        // Checks.
        MathUtils.checkNotNull(f);

        // Reset.
        evaluations = evaluations.withMaximalCount(maxEval).withStart(0);
        T zero = field.getZero();
        T nan  = zero.add(Double.NaN);

        // prepare arrays with the first points
        final T[] x = MathArrays.buildArray(field, maximalOrder + 1);
        final T[] y = MathArrays.buildArray(field, maximalOrder + 1);
        x[0] = min;
        x[1] = startValue;
        x[2] = max;

        // evaluate initial guess
        evaluations.increment();
        y[1] = f.value(x[1]);
        if (Precision.equals(y[1].getReal(), 0.0, 1)) {
            // return the initial guess if it is a perfect root.
            return x[1];
        }

        // evaluate first endpoint
        evaluations.increment();
        y[0] = f.value(x[0]);
        if (Precision.equals(y[0].getReal(), 0.0, 1)) {
            // return the first endpoint if it is a perfect root.
            return x[0];
        }

        int nbPoints;
        int signChangeIndex;
        if (y[0].multiply(y[1]).getReal() < 0) {

            // reduce interval if it brackets the root
            nbPoints        = 2;
            signChangeIndex = 1;

        } else {

            // evaluate second endpoint
            evaluations.increment();
            y[2] = f.value(x[2]);
            if (Precision.equals(y[2].getReal(), 0.0, 1)) {
                // return the second endpoint if it is a perfect root.
                return x[2];
            }

            if (y[1].multiply(y[2]).getReal() < 0) {
                // use all computed point as a start sampling array for solving
                nbPoints        = 3;
                signChangeIndex = 2;
            } else {
                throw new NoBracketingException(x[0].getReal(), x[2].getReal(),
                                                y[0].getReal(), y[2].getReal());
            }

        }

        // prepare a work array for inverse polynomial interpolation
        final T[] tmpX = MathArrays.buildArray(field, x.length);

        // current tightest bracketing of the root
        T xA    = x[signChangeIndex - 1];
        T yA    = y[signChangeIndex - 1];
        T absXA = xA.abs();
        T absYA = yA.abs();
        int agingA   = 0;
        T xB    = x[signChangeIndex];
        T yB    = y[signChangeIndex];
        T absXB = xB.abs();
        T absYB = yB.abs();
        int agingB   = 0;

        // search loop
        while (true) {

            // check convergence of bracketing interval
            T maxX = absXA.subtract(absXB).getReal() < 0 ? absXB : absXA;
            T maxY = absYA.subtract(absYB).getReal() < 0 ? absYB : absYA;
            final T xTol = absoluteAccuracy.add(relativeAccuracy.multiply(maxX));
            if (xB.subtract(xA).subtract(xTol).getReal() <= 0 ||
                maxY.subtract(functionValueAccuracy).getReal() < 0) {
                switch (allowedSolution) {
                case ANY_SIDE :
                    return absYA.subtract(absYB).getReal() < 0 ? xA : xB;
                case LEFT_SIDE :
                    return xA;
                case RIGHT_SIDE :
                    return xB;
                case BELOW_SIDE :
                    return yA.getReal() <= 0 ? xA : xB;
                case ABOVE_SIDE :
                    return yA.getReal() < 0 ? xB : xA;
                default :
                    // this should never happen
                    throw new MathInternalError(null);
                }
            }

            // target for the next evaluation point
            T targetY;
            if (agingA >= MAXIMAL_AGING) {
                // we keep updating the high bracket, try to compensate this
                targetY = yB.divide(16).negate();
            } else if (agingB >= MAXIMAL_AGING) {
                // we keep updating the low bracket, try to compensate this
                targetY = yA.divide(16).negate();
            } else {
                // bracketing is balanced, try to find the root itself
                targetY = zero;
            }

            // make a few attempts to guess a root,
            T nextX;
            int start = 0;
            int end   = nbPoints;
            do {

                // guess a value for current target, using inverse polynomial interpolation
                System.arraycopy(x, start, tmpX, start, end - start);
                nextX = guessX(targetY, tmpX, y, start, end);

                if (!((nextX.subtract(xA).getReal() > 0) && (nextX.subtract(xB).getReal() < 0))) {
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
                    nextX = nan;

                }

            } while (Double.isNaN(nextX.getReal()) && (end - start > 1));

            if (Double.isNaN(nextX.getReal())) {
                // fall back to bisection
                nextX = xA.add(xB.subtract(xA).divide(2));
                start = signChangeIndex - 1;
                end   = signChangeIndex;
            }

            // evaluate the function at the guessed root
            evaluations.increment();
            final T nextY = f.value(nextX);
            if (Precision.equals(nextY.getReal(), 0.0, 1)) {
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
            if (nextY.multiply(yA).getReal() <= 0) {
                // the sign change occurs before the inserted point
                xB = nextX;
                yB = nextY;
                absYB = yB.abs();
                ++agingA;
                agingB = 0;
            } else {
                // the sign change occurs after the inserted point
                xA = nextX;
                yA = nextY;
                absYA = yA.abs();
                agingA = 0;
                ++agingB;

                // update the sign change index
                signChangeIndex++;

            }

        }

    }

    
    private T guessX(final T targetY, final T[] x, final T[] y,
                     final int start, final int end) {

        // compute Q Newton coefficients by divided differences
        for (int i = start; i < end - 1; ++i) {
            final int delta = i + 1 - start;
            for (int j = end - 1; j > i; --j) {
                x[j] = x[j].subtract(x[j-1]).divide(y[j].subtract(y[j - delta]));
            }
        }

        // evaluate Q(targetY)
        T x0 = field.getZero();
        for (int j = end - 1; j >= start; --j) {
            x0 = x[j].add(x0.multiply(targetY.subtract(y[j])));
        }

        return x0;

    }

}
