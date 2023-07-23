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

package org.apache.lucene.util.hnsw.math.special;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class BesselJ
    implements UnivariateFunction {

    // ---------------------------------------------------------------------
    // Mathematical constants
    // ---------------------------------------------------------------------

    
    private static final double PI2 = 0.636619772367581343075535;

    
    private static final double TOWPI1 = 6.28125;

    
    private static final double TWOPI2 = 1.935307179586476925286767e-3;

    
    private static final double TWOPI = TOWPI1 + TWOPI2;

    // ---------------------------------------------------------------------
    // Machine-dependent parameters
    // ---------------------------------------------------------------------

    
    private static final double ENTEN = 1.0e308;

    
    private static final double ENSIG = 1.0e16;

    
    private static final double RTNSIG = 1.0e-4;

    
    private static final double ENMTEN = 8.90e-308;

    
    private static final double X_MIN = 0.0;

    
    private static final double X_MAX = 1.0e4;

    
    private static final double[] FACT = {
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
        3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
        1.307674368e12, 2.0922789888e13, 3.55687428096e14, 6.402373705728e15,
        1.21645100408832e17, 2.43290200817664e18, 5.109094217170944e19,
        1.12400072777760768e21, 2.585201673888497664e22,
        6.2044840173323943936e23
    };

    
    private final double order;

    
    public BesselJ(double order) {
        this.order = order;
    }

    
    public double value(double x)
        throws MathIllegalArgumentException, ConvergenceException {
        return BesselJ.value(order, x);
    }

    
    public static double value(double order, double x)
        throws MathIllegalArgumentException, ConvergenceException {
        final int n = (int) order;
        final double alpha = order - n;
        final int nb = n + 1;
        final BesselJResult res = rjBesl(x, alpha, nb);

        if (res.nVals >= nb) {
            return res.vals[n];
        } else if (res.nVals < 0) {
            throw new MathIllegalArgumentException(LocalizedFormats.BESSEL_FUNCTION_BAD_ARGUMENT,order, x);
        } else if (FastMath.abs(res.vals[res.nVals - 1]) < 1e-100) {
            return res.vals[n]; // underflow; return value (will be zero)
        }
        throw new ConvergenceException(LocalizedFormats.BESSEL_FUNCTION_FAILED_CONVERGENCE, order, x);
    }

    
    public static class BesselJResult {

        
        private final double[] vals;

        
        private final int nVals;

        
        public BesselJResult(double[] b, int n) {
            vals = MathArrays.copyOf(b, b.length);
            nVals = n;
        }

        
        public double[] getVals() {
            return MathArrays.copyOf(vals, vals.length);
        }

        
        public int getnVals() {
            return nVals;
        }
    }

    
    public static BesselJResult rjBesl(double x, double alpha, int nb) {
        final double[] b = new double[nb];

        int ncalc = 0;
        double alpem = 0;
        double alp2em = 0;

        // ---------------------------------------------------------------------
        // Check for out of range arguments.
        // ---------------------------------------------------------------------
        final int magx = (int) x;
        if ((nb > 0) && (x >= X_MIN) && (x <= X_MAX) && (alpha >= 0) &&
            (alpha < 1)) {
            // ---------------------------------------------------------------------
            // Initialize result array to zero.
            // ---------------------------------------------------------------------
            ncalc = nb;
            for (int i = 0; i < nb; ++i) {
                b[i] = 0;
            }

            // ---------------------------------------------------------------------
            // Branch to use 2-term ascending series for small X and asymptotic
            // form for large X when NB is not too large.
            // ---------------------------------------------------------------------
            double tempa;
            double tempb;
            double tempc;
            double tover;
            if (x < RTNSIG) {
                // ---------------------------------------------------------------------
                // Two-term ascending series for small X.
                // ---------------------------------------------------------------------
                tempa = 1;
                alpem = 1 + alpha;
                double halfx = 0;
                if (x > ENMTEN) {
                    halfx = 0.5 * x;
                }
                if (alpha != 0) {
                    tempa = FastMath.pow(halfx, alpha) /
                            (alpha * Gamma.gamma(alpha));
                }
                tempb = 0;
                if (x + 1 > 1) {
                    tempb = -halfx * halfx;
                }
                b[0] = tempa + (tempa * tempb / alpem);
                if ((x != 0) && (b[0] == 0)) {
                    ncalc = 0;
                }
                if (nb != 1) {
                    if (x <= 0) {
                        for (int n = 1; n < nb; ++n) {
                            b[n] = 0;
                        }
                    } else {
                        // ---------------------------------------------------------------------
                        // Calculate higher order functions.
                        // ---------------------------------------------------------------------
                        tempc = halfx;
                        tover = tempb != 0 ? ENMTEN / tempb :  2 * ENMTEN / x;
                        for (int n = 1; n < nb; ++n) {
                            tempa /= alpem;
                            alpem += 1;
                            tempa *= tempc;
                            if (tempa <= tover * alpem) {
                                tempa = 0;
                            }
                            b[n] = tempa + (tempa * tempb / alpem);
                            if ((b[n] == 0) && (ncalc > n)) {
                                ncalc = n;
                            }
                        }
                    }
                }
            } else if ((x > 25.0) && (nb <= magx + 1)) {
                // ---------------------------------------------------------------------
                // Asymptotic series for X > 25
                // ---------------------------------------------------------------------
                final double xc = FastMath.sqrt(PI2 / x);
                final double mul = 0.125 / x;
                final double xin = mul * mul;
                int m = 0;
                if (x >= 130.0) {
                    m = 4;
                } else if (x >= 35.0) {
                    m = 8;
                } else {
                    m = 11;
                }

                final double xm = 4.0 * m;
                // ---------------------------------------------------------------------
                // Argument reduction for SIN and COS routines.
                // ---------------------------------------------------------------------
                double t = (double) ((int) ((x / TWOPI) + 0.5));
                final double z = x - t * TOWPI1 - t * TWOPI2 - (alpha + 0.5) / PI2;
                double vsin = FastMath.sin(z);
                double vcos = FastMath.cos(z);
                double gnu = 2 * alpha;
                double capq;
                double capp;
                double s;
                double t1;
                double xk;
                for (int i = 1; i <= 2; i++) {
                    s = (xm - 1 - gnu) * (xm - 1 + gnu) * xin * 0.5;
                    t = (gnu - (xm - 3.0)) * (gnu + (xm - 3.0));
                    capp = (s * t) / FACT[2 * m];
                    t1 = (gnu - (xm + 1)) * (gnu + (xm + 1));
                    capq = (s * t1) / FACT[2 * m + 1];
                    xk = xm;
                    int k = 2 * m;
                    t1 = t;

                    for (int j = 2; j <= m; j++) {
                        xk -= 4.0;
                        s = (xk - 1 - gnu) * (xk - 1 + gnu);
                        t = (gnu - (xk - 3.0)) * (gnu + (xk - 3.0));
                        capp = (capp + 1 / FACT[k - 2]) * s * t * xin;
                        capq = (capq + 1 / FACT[k - 1]) * s * t1 * xin;
                        k -= 2;
                        t1 = t;
                    }

                    capp += 1;
                    capq = (capq + 1) * ((gnu * gnu) - 1) * (0.125 / x);
                    b[i - 1] = xc * (capp * vcos - capq * vsin);
                    if (nb == 1) {
                        return new BesselJResult(MathArrays.copyOf(b, b.length),
                                                 ncalc);
                    }
                    t = vsin;
                    vsin = -vcos;
                    vcos = t;
                    gnu += 2.0;
                }

                // ---------------------------------------------------------------------
                // If NB > 2, compute J(X,ORDER+I) I = 2, NB-1
                // ---------------------------------------------------------------------
                if (nb > 2) {
                    gnu = 2 * alpha + 2.0;
                    for (int j = 2; j < nb; ++j) {
                        b[j] = gnu * b[j - 1] / x - b[j - 2];
                        gnu += 2.0;
                    }
                }
            } else {
                // ---------------------------------------------------------------------
                // Use recurrence to generate results. First initialize the
                // calculation of P*S.
                // ---------------------------------------------------------------------
                final int nbmx = nb - magx;
                int n = magx + 1;
                int nstart = 0;
                int nend = 0;
                double en = 2 * (n + alpha);
                double plast = 1;
                double p = en / x;
                double pold;
                // ---------------------------------------------------------------------
                // Calculate general significance test.
                // ---------------------------------------------------------------------
                double test = 2 * ENSIG;
                boolean readyToInitialize = false;
                if (nbmx >= 3) {
                    // ---------------------------------------------------------------------
                    // Calculate P*S until N = NB-1. Check for possible
                    // overflow.
                    // ---------------------------------------------------------------------
                    tover = ENTEN / ENSIG;
                    nstart = magx + 2;
                    nend = nb - 1;
                    en = 2 * (nstart - 1 + alpha);
                    double psave;
                    double psavel;
                    for (int k = nstart; k <= nend; k++) {
                        n = k;
                        en += 2.0;
                        pold = plast;
                        plast = p;
                        p = (en * plast / x) - pold;
                        if (p > tover) {
                            // ---------------------------------------------------------------------
                            // To avoid overflow, divide P*S by TOVER. Calculate
                            // P*S until
                            // ABS(P) > 1.
                            // ---------------------------------------------------------------------
                            tover = ENTEN;
                            p /= tover;
                            plast /= tover;
                            psave = p;
                            psavel = plast;
                            nstart = n + 1;
                            do {
                                n += 1;
                                en += 2.0;
                                pold = plast;
                                plast = p;
                                p = (en * plast / x) - pold;
                            } while (p <= 1);
                            tempb = en / x;
                            // ---------------------------------------------------------------------
                            // Calculate backward test and find NCALC, the
                            // highest N such that
                            // the test is passed.
                            // ---------------------------------------------------------------------
                            test = pold * plast * (0.5 - 0.5 / (tempb * tempb));
                            test /= ENSIG;
                            p = plast * tover;
                            n -= 1;
                            en -= 2.0;
                            nend = FastMath.min(nb, n);
                            for (int l = nstart; l <= nend; l++) {
                                pold = psavel;
                                psavel = psave;
                                psave = (en * psavel / x) - pold;
                                if (psave * psavel > test) {
                                    ncalc = l - 1;
                                    readyToInitialize = true;
                                    break;
                                }
                            }
                            ncalc = nend;
                            readyToInitialize = true;
                            break;
                        }
                    }
                    if (!readyToInitialize) {
                        n = nend;
                        en = 2 * (n + alpha);
                        // ---------------------------------------------------------------------
                        // Calculate special significance test for NBMX > 2.
                        // ---------------------------------------------------------------------
                        test = FastMath.max(test, FastMath.sqrt(plast * ENSIG) *
                                                  FastMath.sqrt(2 * p));
                    }
                }
                // ---------------------------------------------------------------------
                // Calculate P*S until significance test passes.
                // ---------------------------------------------------------------------
                if (!readyToInitialize) {
                    do {
                        n += 1;
                        en += 2.0;
                        pold = plast;
                        plast = p;
                        p = (en * plast / x) - pold;
                    } while (p < test);
                }
                // ---------------------------------------------------------------------
                // Initialize the backward recursion and the normalization sum.
                // ---------------------------------------------------------------------
                n += 1;
                en += 2.0;
                tempb = 0;
                tempa = 1 / p;
                int m = (2 * n) - 4 * (n / 2);
                double sum = 0;
                double em = (double) (n / 2);
                alpem = em - 1 + alpha;
                alp2em = 2 * em + alpha;
                if (m != 0) {
                    sum = tempa * alpem * alp2em / em;
                }
                nend = n - nb;

                boolean readyToNormalize = false;
                boolean calculatedB0 = false;

                // ---------------------------------------------------------------------
                // Recur backward via difference equation, calculating (but not
                // storing) B(N), until N = NB.
                // ---------------------------------------------------------------------
                for (int l = 1; l <= nend; l++) {
                    n -= 1;
                    en -= 2.0;
                    tempc = tempb;
                    tempb = tempa;
                    tempa = (en * tempb / x) - tempc;
                    m = 2 - m;
                    if (m != 0) {
                        em -= 1;
                        alp2em = 2 * em + alpha;
                        if (n == 1) {
                            break;
                        }
                        alpem = em - 1 + alpha;
                        if (alpem == 0) {
                            alpem = 1;
                        }
                        sum = (sum + tempa * alp2em) * alpem / em;
                    }
                }

                // ---------------------------------------------------------------------
                // Store B(NB).
                // ---------------------------------------------------------------------
                b[n - 1] = tempa;
                if (nend >= 0) {
                    if (nb <= 1) {
                        alp2em = alpha;
                        if (alpha + 1 == 1) {
                            alp2em = 1;
                        }
                        sum += b[0] * alp2em;
                        readyToNormalize = true;
                    } else {
                        // ---------------------------------------------------------------------
                        // Calculate and store B(NB-1).
                        // ---------------------------------------------------------------------
                        n -= 1;
                        en -= 2.0;
                        b[n - 1] = (en * tempa / x) - tempb;
                        if (n == 1) {
                            calculatedB0 = true;
                        } else {
                            m = 2 - m;
                            if (m != 0) {
                                em -= 1;
                                alp2em = 2 * em + alpha;
                                alpem = em - 1 + alpha;
                                if (alpem == 0) {
                                    alpem = 1;
                                }

                                sum = (sum + (b[n - 1] * alp2em)) * alpem / em;
                            }
                        }
                    }
                }
                if (!readyToNormalize && !calculatedB0) {
                    nend = n - 2;
                    if (nend != 0) {
                        // ---------------------------------------------------------------------
                        // Calculate via difference equation and store B(N),
                        // until N = 2.
                        // ---------------------------------------------------------------------

                        for (int l = 1; l <= nend; l++) {
                            n -= 1;
                            en -= 2.0;
                            b[n - 1] = (en * b[n] / x) - b[n + 1];
                            m = 2 - m;
                            if (m != 0) {
                                em -= 1;
                                alp2em = 2 * em + alpha;
                                alpem = em - 1 + alpha;
                                if (alpem == 0) {
                                    alpem = 1;
                                }

                                sum = (sum + b[n - 1] * alp2em) * alpem / em;
                            }
                        }
                    }
                }
                // ---------------------------------------------------------------------
                // Calculate b[0]
                // ---------------------------------------------------------------------
                if (!readyToNormalize) {
                    if (!calculatedB0) {
                        b[0] = 2.0 * (alpha + 1) * b[1] / x - b[2];
                    }
                    em -= 1;
                    alp2em = 2 * em + alpha;
                    if (alp2em == 0) {
                        alp2em = 1;
                    }
                    sum += b[0] * alp2em;
                }
                // ---------------------------------------------------------------------
                // Normalize. Divide all B(N) by sum.
                // ---------------------------------------------------------------------

                if (FastMath.abs(alpha) > 1e-16) {
                    sum *= Gamma.gamma(alpha) * FastMath.pow(x * 0.5, -alpha);
                }
                tempa = ENMTEN;
                if (sum > 1) {
                    tempa *= sum;
                }

                for (n = 0; n < nb; n++) {
                    if (FastMath.abs(b[n]) < tempa) {
                        b[n] = 0;
                    }
                    b[n] /= sum;
                }
            }
            // ---------------------------------------------------------------------
            // Error return -- X, NB, or ALPHA is out of range.
            // ---------------------------------------------------------------------
        } else {
            if (b.length > 0) {
                b[0] = 0;
            }
            ncalc = FastMath.min(nb, 0) - 1;
        }
        return new BesselJResult(MathArrays.copyOf(b, b.length), ncalc);
    }
}
