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
package org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.noderiv;

import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.MathUnsupportedOperationException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.GoalType;
import org.apache.lucene.util.hnsw.math.optim.PointValuePair;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.LineSearch;
import org.apache.lucene.util.hnsw.math.optim.univariate.UnivariatePointValuePair;


public class PowellOptimizer
    extends MultivariateOptimizer {
    
    private static final double MIN_RELATIVE_TOLERANCE = 2 * FastMath.ulp(1d);
    
    private final double relativeThreshold;
    
    private final double absoluteThreshold;
    
    private final LineSearch line;

    
    public PowellOptimizer(double rel,
                           double abs,
                           ConvergenceChecker<PointValuePair> checker) {
        this(rel, abs, FastMath.sqrt(rel), FastMath.sqrt(abs), checker);
    }

    
    public PowellOptimizer(double rel,
                           double abs,
                           double lineRel,
                           double lineAbs,
                           ConvergenceChecker<PointValuePair> checker) {
        super(checker);

        if (rel < MIN_RELATIVE_TOLERANCE) {
            throw new NumberIsTooSmallException(rel, MIN_RELATIVE_TOLERANCE, true);
        }
        if (abs <= 0) {
            throw new NotStrictlyPositiveException(abs);
        }
        relativeThreshold = rel;
        absoluteThreshold = abs;

        // Create the line search optimizer.
        line = new LineSearch(this,
                              lineRel,
                              lineAbs,
                              1d);
    }

    
    public PowellOptimizer(double rel,
                           double abs) {
        this(rel, abs, null);
    }

    
    public PowellOptimizer(double rel,
                           double abs,
                           double lineRel,
                           double lineAbs) {
        this(rel, abs, lineRel, lineAbs, null);
    }

    
    @Override
    protected PointValuePair doOptimize() {
        checkParameters();

        final GoalType goal = getGoalType();
        final double[] guess = getStartPoint();
        final int n = guess.length;

        final double[][] direc = new double[n][n];
        for (int i = 0; i < n; i++) {
            direc[i][i] = 1;
        }

        final ConvergenceChecker<PointValuePair> checker
            = getConvergenceChecker();

        double[] x = guess;
        double fVal = computeObjectiveValue(x);
        double[] x1 = x.clone();
        while (true) {
            incrementIterationCount();

            double fX = fVal;
            double fX2 = 0;
            double delta = 0;
            int bigInd = 0;
            double alphaMin = 0;

            for (int i = 0; i < n; i++) {
                final double[] d = MathArrays.copyOf(direc[i]);

                fX2 = fVal;

                final UnivariatePointValuePair optimum = line.search(x, d);
                fVal = optimum.getValue();
                alphaMin = optimum.getPoint();
                final double[][] result = newPointAndDirection(x, d, alphaMin);
                x = result[0];

                if ((fX2 - fVal) > delta) {
                    delta = fX2 - fVal;
                    bigInd = i;
                }
            }

            // Default convergence check.
            boolean stop = 2 * (fX - fVal) <=
                (relativeThreshold * (FastMath.abs(fX) + FastMath.abs(fVal)) +
                 absoluteThreshold);

            final PointValuePair previous = new PointValuePair(x1, fX);
            final PointValuePair current = new PointValuePair(x, fVal);
            if (!stop && checker != null) { // User-defined stopping criteria.
                stop = checker.converged(getIterations(), previous, current);
            }
            if (stop) {
                if (goal == GoalType.MINIMIZE) {
                    return (fVal < fX) ? current : previous;
                } else {
                    return (fVal > fX) ? current : previous;
                }
            }

            final double[] d = new double[n];
            final double[] x2 = new double[n];
            for (int i = 0; i < n; i++) {
                d[i] = x[i] - x1[i];
                x2[i] = 2 * x[i] - x1[i];
            }

            x1 = x.clone();
            fX2 = computeObjectiveValue(x2);

            if (fX > fX2) {
                double t = 2 * (fX + fX2 - 2 * fVal);
                double temp = fX - fVal - delta;
                t *= temp * temp;
                temp = fX - fX2;
                t -= delta * temp * temp;

                if (t < 0.0) {
                    final UnivariatePointValuePair optimum = line.search(x, d);
                    fVal = optimum.getValue();
                    alphaMin = optimum.getPoint();
                    final double[][] result = newPointAndDirection(x, d, alphaMin);
                    x = result[0];

                    final int lastInd = n - 1;
                    direc[bigInd] = direc[lastInd];
                    direc[lastInd] = result[1];
                }
            }
        }
    }

    
    private double[][] newPointAndDirection(double[] p,
                                            double[] d,
                                            double optimum) {
        final int n = p.length;
        final double[] nP = new double[n];
        final double[] nD = new double[n];
        for (int i = 0; i < n; i++) {
            nD[i] = d[i] * optimum;
            nP[i] = p[i] + nD[i];
        }

        final double[][] result = new double[2][];
        result[0] = nP;
        result[1] = nD;

        return result;
    }

    
    private void checkParameters() {
        if (getLowerBound() != null ||
            getUpperBound() != null) {
            throw new MathUnsupportedOperationException(LocalizedFormats.CONSTRAINT);
        }
    }
}
