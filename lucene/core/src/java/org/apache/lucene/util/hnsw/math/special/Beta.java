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

import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.util.ContinuedFraction;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class Beta {
    
    private static final double DEFAULT_EPSILON = 1E-14;

    
    private static final double HALF_LOG_TWO_PI = .9189385332046727;

    
    private static final double[] DELTA = {
        .833333333333333333333333333333E-01,
        -.277777777777777777777777752282E-04,
        .793650793650793650791732130419E-07,
        -.595238095238095232389839236182E-09,
        .841750841750832853294451671990E-11,
        -.191752691751854612334149171243E-12,
        .641025640510325475730918472625E-14,
        -.295506514125338232839867823991E-15,
        .179643716359402238723287696452E-16,
        -.139228964661627791231203060395E-17,
        .133802855014020915603275339093E-18,
        -.154246009867966094273710216533E-19,
        .197701992980957427278370133333E-20,
        -.234065664793997056856992426667E-21,
        .171348014966398575409015466667E-22
    };

    
    private Beta() {}

    
    public static double regularizedBeta(double x, double a, double b) {
        return regularizedBeta(x, a, b, DEFAULT_EPSILON, Integer.MAX_VALUE);
    }

    
    public static double regularizedBeta(double x,
                                         double a, double b,
                                         double epsilon) {
        return regularizedBeta(x, a, b, epsilon, Integer.MAX_VALUE);
    }

    
    public static double regularizedBeta(double x,
                                         double a, double b,
                                         int maxIterations) {
        return regularizedBeta(x, a, b, DEFAULT_EPSILON, maxIterations);
    }

    
    public static double regularizedBeta(double x,
                                         final double a, final double b,
                                         double epsilon, int maxIterations) {
        double ret;

        if (Double.isNaN(x) ||
            Double.isNaN(a) ||
            Double.isNaN(b) ||
            x < 0 ||
            x > 1 ||
            a <= 0 ||
            b <= 0) {
            ret = Double.NaN;
        } else if (x > (a + 1) / (2 + b + a) &&
                   1 - x <= (b + 1) / (2 + b + a)) {
            ret = 1 - regularizedBeta(1 - x, b, a, epsilon, maxIterations);
        } else {
            ContinuedFraction fraction = new ContinuedFraction() {

                
                @Override
                protected double getB(int n, double x) {
                    double ret;
                    double m;
                    if (n % 2 == 0) { // even
                        m = n / 2.0;
                        ret = (m * (b - m) * x) /
                            ((a + (2 * m) - 1) * (a + (2 * m)));
                    } else {
                        m = (n - 1.0) / 2.0;
                        ret = -((a + m) * (a + b + m) * x) /
                                ((a + (2 * m)) * (a + (2 * m) + 1.0));
                    }
                    return ret;
                }

                
                @Override
                protected double getA(int n, double x) {
                    return 1.0;
                }
            };
            ret = FastMath.exp((a * FastMath.log(x)) + (b * FastMath.log1p(-x)) -
                FastMath.log(a) - logBeta(a, b)) *
                1.0 / fraction.evaluate(x, epsilon, maxIterations);
        }

        return ret;
    }

    
    @Deprecated
    public static double logBeta(double a, double b,
                                 double epsilon,
                                 int maxIterations) {

        return logBeta(a, b);
    }


    
    private static double logGammaSum(final double a, final double b)
        throws OutOfRangeException {

        if ((a < 1.0) || (a > 2.0)) {
            throw new OutOfRangeException(a, 1.0, 2.0);
        }
        if ((b < 1.0) || (b > 2.0)) {
            throw new OutOfRangeException(b, 1.0, 2.0);
        }

        final double x = (a - 1.0) + (b - 1.0);
        if (x <= 0.5) {
            return Gamma.logGamma1p(1.0 + x);
        } else if (x <= 1.5) {
            return Gamma.logGamma1p(x) + FastMath.log1p(x);
        } else {
            return Gamma.logGamma1p(x - 1.0) + FastMath.log(x * (1.0 + x));
        }
    }

    
    private static double logGammaMinusLogGammaSum(final double a,
                                                   final double b)
        throws NumberIsTooSmallException {

        if (a < 0.0) {
            throw new NumberIsTooSmallException(a, 0.0, true);
        }
        if (b < 10.0) {
            throw new NumberIsTooSmallException(b, 10.0, true);
        }

        /*
         * d = a + b - 0.5
         */
        final double d;
        final double w;
        if (a <= b) {
            d = b + (a - 0.5);
            w = deltaMinusDeltaSum(a, b);
        } else {
            d = a + (b - 0.5);
            w = deltaMinusDeltaSum(b, a);
        }

        final double u = d * FastMath.log1p(a / b);
        final double v = a * (FastMath.log(b) - 1.0);

        return u <= v ? (w - u) - v : (w - v) - u;
    }

    
    private static double deltaMinusDeltaSum(final double a,
                                             final double b)
        throws OutOfRangeException, NumberIsTooSmallException {

        if ((a < 0) || (a > b)) {
            throw new OutOfRangeException(a, 0, b);
        }
        if (b < 10) {
            throw new NumberIsTooSmallException(b, 10, true);
        }

        final double h = a / b;
        final double p = h / (1.0 + h);
        final double q = 1.0 / (1.0 + h);
        final double q2 = q * q;
        /*
         * s[i] = 1 + q + ... - q**(2 * i)
         */
        final double[] s = new double[DELTA.length];
        s[0] = 1.0;
        for (int i = 1; i < s.length; i++) {
            s[i] = 1.0 + (q + q2 * s[i - 1]);
        }
        /*
         * w = Delta(b) - Delta(a + b)
         */
        final double sqrtT = 10.0 / b;
        final double t = sqrtT * sqrtT;
        double w = DELTA[DELTA.length - 1] * s[s.length - 1];
        for (int i = DELTA.length - 2; i >= 0; i--) {
            w = t * w + DELTA[i] * s[i];
        }
        return w * p / b;
    }

    
    private static double sumDeltaMinusDeltaSum(final double p,
                                                final double q) {

        if (p < 10.0) {
            throw new NumberIsTooSmallException(p, 10.0, true);
        }
        if (q < 10.0) {
            throw new NumberIsTooSmallException(q, 10.0, true);
        }

        final double a = FastMath.min(p, q);
        final double b = FastMath.max(p, q);
        final double sqrtT = 10.0 / a;
        final double t = sqrtT * sqrtT;
        double z = DELTA[DELTA.length - 1];
        for (int i = DELTA.length - 2; i >= 0; i--) {
            z = t * z + DELTA[i];
        }
        return z / a + deltaMinusDeltaSum(a, b);
    }

    
    public static double logBeta(final double p, final double q) {
        if (Double.isNaN(p) || Double.isNaN(q) || (p <= 0.0) || (q <= 0.0)) {
            return Double.NaN;
        }

        final double a = FastMath.min(p, q);
        final double b = FastMath.max(p, q);
        if (a >= 10.0) {
            final double w = sumDeltaMinusDeltaSum(a, b);
            final double h = a / b;
            final double c = h / (1.0 + h);
            final double u = -(a - 0.5) * FastMath.log(c);
            final double v = b * FastMath.log1p(h);
            if (u <= v) {
                return (((-0.5 * FastMath.log(b) + HALF_LOG_TWO_PI) + w) - u) - v;
            } else {
                return (((-0.5 * FastMath.log(b) + HALF_LOG_TWO_PI) + w) - v) - u;
            }
        } else if (a > 2.0) {
            if (b > 1000.0) {
                final int n = (int) FastMath.floor(a - 1.0);
                double prod = 1.0;
                double ared = a;
                for (int i = 0; i < n; i++) {
                    ared -= 1.0;
                    prod *= ared / (1.0 + ared / b);
                }
                return (FastMath.log(prod) - n * FastMath.log(b)) +
                        (Gamma.logGamma(ared) +
                         logGammaMinusLogGammaSum(ared, b));
            } else {
                double prod1 = 1.0;
                double ared = a;
                while (ared > 2.0) {
                    ared -= 1.0;
                    final double h = ared / b;
                    prod1 *= h / (1.0 + h);
                }
                if (b < 10.0) {
                    double prod2 = 1.0;
                    double bred = b;
                    while (bred > 2.0) {
                        bred -= 1.0;
                        prod2 *= bred / (ared + bred);
                    }
                    return FastMath.log(prod1) +
                           FastMath.log(prod2) +
                           (Gamma.logGamma(ared) +
                           (Gamma.logGamma(bred) -
                            logGammaSum(ared, bred)));
                } else {
                    return FastMath.log(prod1) +
                           Gamma.logGamma(ared) +
                           logGammaMinusLogGammaSum(ared, b);
                }
            }
        } else if (a >= 1.0) {
            if (b > 2.0) {
                if (b < 10.0) {
                    double prod = 1.0;
                    double bred = b;
                    while (bred > 2.0) {
                        bred -= 1.0;
                        prod *= bred / (a + bred);
                    }
                    return FastMath.log(prod) +
                           (Gamma.logGamma(a) +
                            (Gamma.logGamma(bred) -
                             logGammaSum(a, bred)));
                } else {
                    return Gamma.logGamma(a) +
                           logGammaMinusLogGammaSum(a, b);
                }
            } else {
                return Gamma.logGamma(a) +
                       Gamma.logGamma(b) -
                       logGammaSum(a, b);
            }
        } else {
            if (b >= 10.0) {
                return Gamma.logGamma(a) +
                       logGammaMinusLogGammaSum(a, b);
            } else {
                // The following command is the original NSWC implementation.
                // return Gamma.logGamma(a) +
                // (Gamma.logGamma(b) - Gamma.logGamma(a + b));
                // The following command turns out to be more accurate.
                return FastMath.log(Gamma.gamma(a) * Gamma.gamma(b) /
                                    Gamma.gamma(a + b));
            }
        }
    }
}
