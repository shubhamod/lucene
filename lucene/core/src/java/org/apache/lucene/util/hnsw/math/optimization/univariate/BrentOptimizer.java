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
package org.apache.lucene.util.hnsw.math.optimization.univariate;

import org.apache.lucene.util.hnsw.math.util.Precision;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optimization.GoalType;


@Deprecated
public class BrentOptimizer extends BaseAbstractUnivariateOptimizer {
    
    private static final double GOLDEN_SECTION = 0.5 * (3 - FastMath.sqrt(5));
    
    private static final double MIN_RELATIVE_TOLERANCE = 2 * FastMath.ulp(1d);
    
    private final double relativeThreshold;
    
    private final double absoluteThreshold;

    
    public BrentOptimizer(double rel,
                          double abs,
                          ConvergenceChecker<UnivariatePointValuePair> checker) {
        super(checker);

        if (rel < MIN_RELATIVE_TOLERANCE) {
            throw new NumberIsTooSmallException(rel, MIN_RELATIVE_TOLERANCE, true);
        }
        if (abs <= 0) {
            throw new NotStrictlyPositiveException(abs);
        }

        relativeThreshold = rel;
        absoluteThreshold = abs;
    }

    
    public BrentOptimizer(double rel,
                          double abs) {
        this(rel, abs, null);
    }

    
    @Override
    protected UnivariatePointValuePair doOptimize() {
        final boolean isMinim = getGoalType() == GoalType.MINIMIZE;
        final double lo = getMin();
        final double mid = getStartValue();
        final double hi = getMax();

        // Optional additional convergence criteria.
        final ConvergenceChecker<UnivariatePointValuePair> checker
            = getConvergenceChecker();

        double a;
        double b;
        if (lo < hi) {
            a = lo;
            b = hi;
        } else {
            a = hi;
            b = lo;
        }

        double x = mid;
        double v = x;
        double w = x;
        double d = 0;
        double e = 0;
        double fx = computeObjectiveValue(x);
        if (!isMinim) {
            fx = -fx;
        }
        double fv = fx;
        double fw = fx;

        UnivariatePointValuePair previous = null;
        UnivariatePointValuePair current
            = new UnivariatePointValuePair(x, isMinim ? fx : -fx);
        // Best point encountered so far (which is the initial guess).
        UnivariatePointValuePair best = current;

        int iter = 0;
        while (true) {
            final double m = 0.5 * (a + b);
            final double tol1 = relativeThreshold * FastMath.abs(x) + absoluteThreshold;
            final double tol2 = 2 * tol1;

            // Default stopping criterion.
            final boolean stop = FastMath.abs(x - m) <= tol2 - 0.5 * (b - a);
            if (!stop) {
                double p = 0;
                double q = 0;
                double r = 0;
                double u = 0;

                if (FastMath.abs(e) > tol1) { // Fit parabola.
                    r = (x - w) * (fx - fv);
                    q = (x - v) * (fx - fw);
                    p = (x - v) * q - (x - w) * r;
                    q = 2 * (q - r);

                    if (q > 0) {
                        p = -p;
                    } else {
                        q = -q;
                    }

                    r = e;
                    e = d;

                    if (p > q * (a - x) &&
                        p < q * (b - x) &&
                        FastMath.abs(p) < FastMath.abs(0.5 * q * r)) {
                        // Parabolic interpolation step.
                        d = p / q;
                        u = x + d;

                        // f must not be evaluated too close to a or b.
                        if (u - a < tol2 || b - u < tol2) {
                            if (x <= m) {
                                d = tol1;
                            } else {
                                d = -tol1;
                            }
                        }
                    } else {
                        // Golden section step.
                        if (x < m) {
                            e = b - x;
                        } else {
                            e = a - x;
                        }
                        d = GOLDEN_SECTION * e;
                    }
                } else {
                    // Golden section step.
                    if (x < m) {
                        e = b - x;
                    } else {
                        e = a - x;
                    }
                    d = GOLDEN_SECTION * e;
                }

                // Update by at least "tol1".
                if (FastMath.abs(d) < tol1) {
                    if (d >= 0) {
                        u = x + tol1;
                    } else {
                        u = x - tol1;
                    }
                } else {
                    u = x + d;
                }

                double fu = computeObjectiveValue(u);
                if (!isMinim) {
                    fu = -fu;
                }

                // User-defined convergence checker.
                previous = current;
                current = new UnivariatePointValuePair(u, isMinim ? fu : -fu);
                best = best(best,
                            best(previous,
                                 current,
                                 isMinim),
                            isMinim);

                if (checker != null && checker.converged(iter, previous, current)) {
                    return best;
                }

                // Update a, b, v, w and x.
                if (fu <= fx) {
                    if (u < x) {
                        b = x;
                    } else {
                        a = x;
                    }
                    v = w;
                    fv = fw;
                    w = x;
                    fw = fx;
                    x = u;
                    fx = fu;
                } else {
                    if (u < x) {
                        a = u;
                    } else {
                        b = u;
                    }
                    if (fu <= fw ||
                        Precision.equals(w, x)) {
                        v = w;
                        fv = fw;
                        w = u;
                        fw = fu;
                    } else if (fu <= fv ||
                               Precision.equals(v, x) ||
                               Precision.equals(v, w)) {
                        v = u;
                        fv = fu;
                    }
                }
            } else { // Default termination (Brent's criterion).
                return best(best,
                            best(previous,
                                 current,
                                 isMinim),
                            isMinim);
            }
            ++iter;
        }
    }

    
    private UnivariatePointValuePair best(UnivariatePointValuePair a,
                                          UnivariatePointValuePair b,
                                          boolean isMinim) {
        if (a == null) {
            return b;
        }
        if (b == null) {
            return a;
        }

        if (isMinim) {
            return a.getValue() <= b.getValue() ? a : b;
        } else {
            return a.getValue() >= b.getValue() ? a : b;
        }
    }
}
