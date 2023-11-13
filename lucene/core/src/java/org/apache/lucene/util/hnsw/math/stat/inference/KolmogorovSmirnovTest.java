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

package org.apache.lucene.util.hnsw.math.stat.inference;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.HashSet;

import org.apache.lucene.util.hnsw.math.distribution.EnumeratedRealDistribution;
import org.apache.lucene.util.hnsw.math.distribution.RealDistribution;
import org.apache.lucene.util.hnsw.math.distribution.UniformRealDistribution;
import org.apache.lucene.util.hnsw.math.exception.InsufficientDataException;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.TooManyIterationsException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.fraction.BigFraction;
import org.apache.lucene.util.hnsw.math.fraction.BigFractionField;
import org.apache.lucene.util.hnsw.math.fraction.FractionConversionException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowFieldMatrix;
import org.apache.lucene.util.hnsw.math.linear.FieldMatrix;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.random.JDKRandomGenerator;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.CombinatoricsUtils;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class KolmogorovSmirnovTest {

    
    protected static final int MAXIMUM_PARTIAL_SUM_COUNT = 100000;

    
    protected static final double KS_SUM_CAUCHY_CRITERION = 1E-20;

    
    protected static final double PG_SUM_RELATIVE_ERROR = 1.0e-10;

    
    @Deprecated
    protected static final int SMALL_SAMPLE_PRODUCT = 200;

    
    protected static final int LARGE_SAMPLE_PRODUCT = 10000;

    
    @Deprecated
    protected static final int MONTE_CARLO_ITERATIONS = 1000000;

    
    private final RandomGenerator rng;

    
    public KolmogorovSmirnovTest() {
        rng = new Well19937c();
    }

    
    @Deprecated
    public KolmogorovSmirnovTest(RandomGenerator rng) {
        this.rng = rng;
    }

    
    public double kolmogorovSmirnovTest(RealDistribution distribution, double[] data, boolean exact) {
        return 1d - cdf(kolmogorovSmirnovStatistic(distribution, data), data.length, exact);
    }

    
    public double kolmogorovSmirnovStatistic(RealDistribution distribution, double[] data) {
        checkArray(data);
        final int n = data.length;
        final double nd = n;
        final double[] dataCopy = new double[n];
        System.arraycopy(data, 0, dataCopy, 0, n);
        Arrays.sort(dataCopy);
        double d = 0d;
        for (int i = 1; i <= n; i++) {
            final double yi = distribution.cumulativeProbability(dataCopy[i - 1]);
            final double currD = FastMath.max(yi - (i - 1) / nd, i / nd - yi);
            if (currD > d) {
                d = currD;
            }
        }
        return d;
    }

    
    public double kolmogorovSmirnovTest(double[] x, double[] y, boolean strict) {
        final long lengthProduct = (long) x.length * y.length;
        double[] xa = null;
        double[] ya = null;
        if (lengthProduct < LARGE_SAMPLE_PRODUCT && hasTies(x,y)) {
            xa = MathArrays.copyOf(x);
            ya = MathArrays.copyOf(y);
            fixTies(xa, ya);
        } else {
            xa = x;
            ya = y;
        }
        if (lengthProduct < LARGE_SAMPLE_PRODUCT) {
            return exactP(kolmogorovSmirnovStatistic(xa, ya), x.length, y.length, strict);
        }
        return approximateP(kolmogorovSmirnovStatistic(x, y), x.length, y.length);
    }

    
    public double kolmogorovSmirnovTest(double[] x, double[] y) {
        return kolmogorovSmirnovTest(x, y, true);
    }

    
    public double kolmogorovSmirnovStatistic(double[] x, double[] y) {
        return integralKolmogorovSmirnovStatistic(x, y)/((double)(x.length * (long)y.length));
    }

    
    private long integralKolmogorovSmirnovStatistic(double[] x, double[] y) {
        checkArray(x);
        checkArray(y);
        // Copy and sort the sample arrays
        final double[] sx = MathArrays.copyOf(x);
        final double[] sy = MathArrays.copyOf(y);
        Arrays.sort(sx);
        Arrays.sort(sy);
        final int n = sx.length;
        final int m = sy.length;

        int rankX = 0;
        int rankY = 0;
        long curD = 0l;

        // Find the max difference between cdf_x and cdf_y
        long supD = 0l;
        do {
            double z = Double.compare(sx[rankX], sy[rankY]) <= 0 ? sx[rankX] : sy[rankY];
            while(rankX < n && Double.compare(sx[rankX], z) == 0) {
                rankX += 1;
                curD += m;
            }
            while(rankY < m && Double.compare(sy[rankY], z) == 0) {
                rankY += 1;
                curD -= n;
            }
            if (curD > supD) {
                supD = curD;
            }
            else if (-curD > supD) {
                supD = -curD;
            }
        } while(rankX < n && rankY < m);
        return supD;
    }

    
    public double kolmogorovSmirnovTest(RealDistribution distribution, double[] data) {
        return kolmogorovSmirnovTest(distribution, data, false);
    }

    
    public boolean kolmogorovSmirnovTest(RealDistribution distribution, double[] data, double alpha) {
        if ((alpha <= 0) || (alpha > 0.5)) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_BOUND_SIGNIFICANCE_LEVEL, alpha, 0, 0.5);
        }
        return kolmogorovSmirnovTest(distribution, data) < alpha;
    }

    
    public double bootstrap(double[] x, double[] y, int iterations, boolean strict) {
        final int xLength = x.length;
        final int yLength = y.length;
        final double[] combined = new double[xLength + yLength];
        System.arraycopy(x, 0, combined, 0, xLength);
        System.arraycopy(y, 0, combined, xLength, yLength);
        final EnumeratedRealDistribution dist = new EnumeratedRealDistribution(rng, combined);
        final long d = integralKolmogorovSmirnovStatistic(x, y);
        int greaterCount = 0;
        int equalCount = 0;
        double[] curX;
        double[] curY;
        long curD;
        for (int i = 0; i < iterations; i++) {
            curX = dist.sample(xLength);
            curY = dist.sample(yLength);
            curD = integralKolmogorovSmirnovStatistic(curX, curY);
            if (curD > d) {
                greaterCount++;
            } else if (curD == d) {
                equalCount++;
            }
        }
        return strict ? greaterCount / (double) iterations :
            (greaterCount + equalCount) / (double) iterations;
    }

    
    public double bootstrap(double[] x, double[] y, int iterations) {
        return bootstrap(x, y, iterations, true);
    }

    
    public double cdf(double d, int n)
        throws MathArithmeticException {
        return cdf(d, n, false);
    }

    
    public double cdfExact(double d, int n)
        throws MathArithmeticException {
        return cdf(d, n, true);
    }

    
    public double cdf(double d, int n, boolean exact)
        throws MathArithmeticException {

        final double ninv = 1 / ((double) n);
        final double ninvhalf = 0.5 * ninv;

        if (d <= ninvhalf) {
            return 0;
        } else if (ninvhalf < d && d <= ninv) {
            double res = 1;
            final double f = 2 * d - ninv;
            // n! f^n = n*f * (n-1)*f * ... * 1*x
            for (int i = 1; i <= n; ++i) {
                res *= i * f;
            }
            return res;
        } else if (1 - ninv <= d && d < 1) {
            return 1 - 2 * Math.pow(1 - d, n);
        } else if (1 <= d) {
            return 1;
        }
        if (exact) {
            return exactK(d, n);
        }
        if (n <= 140) {
            return roundedK(d, n);
        }
        return pelzGood(d, n);
    }

    
    private double exactK(double d, int n)
        throws MathArithmeticException {

        final int k = (int) Math.ceil(n * d);

        final FieldMatrix<BigFraction> H = this.createExactH(d, n);
        final FieldMatrix<BigFraction> Hpower = H.power(n);

        BigFraction pFrac = Hpower.getEntry(k - 1, k - 1);

        for (int i = 1; i <= n; ++i) {
            pFrac = pFrac.multiply(i).divide(n);
        }

        /*
         * BigFraction.doubleValue converts numerator to double and the denominator to double and
         * divides afterwards. That gives NaN quite easy. This does not (scale is the number of
         * digits):
         */
        return pFrac.bigDecimalValue(20, BigDecimal.ROUND_HALF_UP).doubleValue();
    }

    
    private double roundedK(double d, int n) {

        final int k = (int) Math.ceil(n * d);
        final RealMatrix H = this.createRoundedH(d, n);
        final RealMatrix Hpower = H.power(n);

        double pFrac = Hpower.getEntry(k - 1, k - 1);
        for (int i = 1; i <= n; ++i) {
            pFrac *= (double) i / (double) n;
        }

        return pFrac;
    }

    
    public double pelzGood(double d, int n) {
        // Change the variable since approximation is for the distribution evaluated at d / sqrt(n)
        final double sqrtN = FastMath.sqrt(n);
        final double z = d * sqrtN;
        final double z2 = d * d * n;
        final double z4 = z2 * z2;
        final double z6 = z4 * z2;
        final double z8 = z4 * z4;

        // Eventual return value
        double ret = 0;

        // Compute K_0(z)
        double sum = 0;
        double increment = 0;
        double kTerm = 0;
        double z2Term = MathUtils.PI_SQUARED / (8 * z2);
        int k = 1;
        for (; k < MAXIMUM_PARTIAL_SUM_COUNT; k++) {
            kTerm = 2 * k - 1;
            increment = FastMath.exp(-z2Term * kTerm * kTerm);
            sum += increment;
            if (increment <= PG_SUM_RELATIVE_ERROR * sum) {
                break;
            }
        }
        if (k == MAXIMUM_PARTIAL_SUM_COUNT) {
            throw new TooManyIterationsException(MAXIMUM_PARTIAL_SUM_COUNT);
        }
        ret = sum * FastMath.sqrt(2 * FastMath.PI) / z;

        // K_1(z)
        // Sum is -inf to inf, but k term is always (k + 1/2) ^ 2, so really have
        // twice the sum from k = 0 to inf (k = -1 is same as 0, -2 same as 1, ...)
        final double twoZ2 = 2 * z2;
        sum = 0;
        kTerm = 0;
        double kTerm2 = 0;
        for (k = 0; k < MAXIMUM_PARTIAL_SUM_COUNT; k++) {
            kTerm = k + 0.5;
            kTerm2 = kTerm * kTerm;
            increment = (MathUtils.PI_SQUARED * kTerm2 - z2) * FastMath.exp(-MathUtils.PI_SQUARED * kTerm2 / twoZ2);
            sum += increment;
            if (FastMath.abs(increment) < PG_SUM_RELATIVE_ERROR * FastMath.abs(sum)) {
                break;
            }
        }
        if (k == MAXIMUM_PARTIAL_SUM_COUNT) {
            throw new TooManyIterationsException(MAXIMUM_PARTIAL_SUM_COUNT);
        }
        final double sqrtHalfPi = FastMath.sqrt(FastMath.PI / 2);
        // Instead of doubling sum, divide by 3 instead of 6
        ret += sum * sqrtHalfPi / (3 * z4 * sqrtN);

        // K_2(z)
        // Same drill as K_1, but with two doubly infinite sums, all k terms are even powers.
        final double z4Term = 2 * z4;
        final double z6Term = 6 * z6;
        z2Term = 5 * z2;
        final double pi4 = MathUtils.PI_SQUARED * MathUtils.PI_SQUARED;
        sum = 0;
        kTerm = 0;
        kTerm2 = 0;
        for (k = 0; k < MAXIMUM_PARTIAL_SUM_COUNT; k++) {
            kTerm = k + 0.5;
            kTerm2 = kTerm * kTerm;
            increment =  (z6Term + z4Term + MathUtils.PI_SQUARED * (z4Term - z2Term) * kTerm2 +
                    pi4 * (1 - twoZ2) * kTerm2 * kTerm2) * FastMath.exp(-MathUtils.PI_SQUARED * kTerm2 / twoZ2);
            sum += increment;
            if (FastMath.abs(increment) < PG_SUM_RELATIVE_ERROR * FastMath.abs(sum)) {
                break;
            }
        }
        if (k == MAXIMUM_PARTIAL_SUM_COUNT) {
            throw new TooManyIterationsException(MAXIMUM_PARTIAL_SUM_COUNT);
        }
        double sum2 = 0;
        kTerm2 = 0;
        for (k = 1; k < MAXIMUM_PARTIAL_SUM_COUNT; k++) {
            kTerm2 = k * k;
            increment = MathUtils.PI_SQUARED * kTerm2 * FastMath.exp(-MathUtils.PI_SQUARED * kTerm2 / twoZ2);
            sum2 += increment;
            if (FastMath.abs(increment) < PG_SUM_RELATIVE_ERROR * FastMath.abs(sum2)) {
                break;
            }
        }
        if (k == MAXIMUM_PARTIAL_SUM_COUNT) {
            throw new TooManyIterationsException(MAXIMUM_PARTIAL_SUM_COUNT);
        }
        // Again, adjust coefficients instead of doubling sum, sum2
        ret += (sqrtHalfPi / n) * (sum / (36 * z2 * z2 * z2 * z) - sum2 / (18 * z2 * z));

        // K_3(z) One more time with feeling - two doubly infinite sums, all k powers even.
        // Multiply coefficient denominators by 2, so omit doubling sums.
        final double pi6 = pi4 * MathUtils.PI_SQUARED;
        sum = 0;
        double kTerm4 = 0;
        double kTerm6 = 0;
        for (k = 0; k < MAXIMUM_PARTIAL_SUM_COUNT; k++) {
            kTerm = k + 0.5;
            kTerm2 = kTerm * kTerm;
            kTerm4 = kTerm2 * kTerm2;
            kTerm6 = kTerm4 * kTerm2;
            increment = (pi6 * kTerm6 * (5 - 30 * z2) + pi4 * kTerm4 * (-60 * z2 + 212 * z4) +
                            MathUtils.PI_SQUARED * kTerm2 * (135 * z4 - 96 * z6) - 30 * z6 - 90 * z8) *
                    FastMath.exp(-MathUtils.PI_SQUARED * kTerm2 / twoZ2);
            sum += increment;
            if (FastMath.abs(increment) < PG_SUM_RELATIVE_ERROR * FastMath.abs(sum)) {
                break;
            }
        }
        if (k == MAXIMUM_PARTIAL_SUM_COUNT) {
            throw new TooManyIterationsException(MAXIMUM_PARTIAL_SUM_COUNT);
        }
        sum2 = 0;
        for (k = 1; k < MAXIMUM_PARTIAL_SUM_COUNT; k++) {
            kTerm2 = k * k;
            kTerm4 = kTerm2 * kTerm2;
            increment = (-pi4 * kTerm4 + 3 * MathUtils.PI_SQUARED * kTerm2 * z2) *
                    FastMath.exp(-MathUtils.PI_SQUARED * kTerm2 / twoZ2);
            sum2 += increment;
            if (FastMath.abs(increment) < PG_SUM_RELATIVE_ERROR * FastMath.abs(sum2)) {
                break;
            }
        }
        if (k == MAXIMUM_PARTIAL_SUM_COUNT) {
            throw new TooManyIterationsException(MAXIMUM_PARTIAL_SUM_COUNT);
        }
        return ret + (sqrtHalfPi / (sqrtN * n)) * (sum / (3240 * z6 * z4) +
                + sum2 / (108 * z6));

    }

    
    private FieldMatrix<BigFraction> createExactH(double d, int n)
        throws NumberIsTooLargeException, FractionConversionException {

        final int k = (int) Math.ceil(n * d);
        final int m = 2 * k - 1;
        final double hDouble = k - n * d;
        if (hDouble >= 1) {
            throw new NumberIsTooLargeException(hDouble, 1.0, false);
        }
        BigFraction h = null;
        try {
            h = new BigFraction(hDouble, 1.0e-20, 10000);
        } catch (final FractionConversionException e1) {
            try {
                h = new BigFraction(hDouble, 1.0e-10, 10000);
            } catch (final FractionConversionException e2) {
                h = new BigFraction(hDouble, 1.0e-5, 10000);
            }
        }
        final BigFraction[][] Hdata = new BigFraction[m][m];

        /*
         * Start by filling everything with either 0 or 1.
         */
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                if (i - j + 1 < 0) {
                    Hdata[i][j] = BigFraction.ZERO;
                } else {
                    Hdata[i][j] = BigFraction.ONE;
                }
            }
        }

        /*
         * Setting up power-array to avoid calculating the same value twice: hPowers[0] = h^1 ...
         * hPowers[m-1] = h^m
         */
        final BigFraction[] hPowers = new BigFraction[m];
        hPowers[0] = h;
        for (int i = 1; i < m; ++i) {
            hPowers[i] = h.multiply(hPowers[i - 1]);
        }

        /*
         * First column and last row has special values (each other reversed).
         */
        for (int i = 0; i < m; ++i) {
            Hdata[i][0] = Hdata[i][0].subtract(hPowers[i]);
            Hdata[m - 1][i] = Hdata[m - 1][i].subtract(hPowers[m - i - 1]);
        }

        /*
         * [1] states: "For 1/2 < h < 1 the bottom left element of the matrix should be (1 - 2*h^m +
         * (2h - 1)^m )/m!" Since 0 <= h < 1, then if h > 1/2 is sufficient to check:
         */
        if (h.compareTo(BigFraction.ONE_HALF) == 1) {
            Hdata[m - 1][0] = Hdata[m - 1][0].add(h.multiply(2).subtract(1).pow(m));
        }

        /*
         * Aside from the first column and last row, the (i, j)-th element is 1/(i - j + 1)! if i -
         * j + 1 >= 0, else 0. 1's and 0's are already put, so only division with (i - j + 1)! is
         * needed in the elements that have 1's. There is no need to calculate (i - j + 1)! and then
         * divide - small steps avoid overflows. Note that i - j + 1 > 0 <=> i + 1 > j instead of
         * j'ing all the way to m. Also note that it is started at g = 2 because dividing by 1 isn't
         * really necessary.
         */
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                if (i - j + 1 > 0) {
                    for (int g = 2; g <= i - j + 1; ++g) {
                        Hdata[i][j] = Hdata[i][j].divide(g);
                    }
                }
            }
        }
        return new Array2DRowFieldMatrix<BigFraction>(BigFractionField.getInstance(), Hdata);
    }

    
    private RealMatrix createRoundedH(double d, int n)
        throws NumberIsTooLargeException {

        final int k = (int) Math.ceil(n * d);
        final int m = 2 * k - 1;
        final double h = k - n * d;
        if (h >= 1) {
            throw new NumberIsTooLargeException(h, 1.0, false);
        }
        final double[][] Hdata = new double[m][m];

        /*
         * Start by filling everything with either 0 or 1.
         */
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                if (i - j + 1 < 0) {
                    Hdata[i][j] = 0;
                } else {
                    Hdata[i][j] = 1;
                }
            }
        }

        /*
         * Setting up power-array to avoid calculating the same value twice: hPowers[0] = h^1 ...
         * hPowers[m-1] = h^m
         */
        final double[] hPowers = new double[m];
        hPowers[0] = h;
        for (int i = 1; i < m; ++i) {
            hPowers[i] = h * hPowers[i - 1];
        }

        /*
         * First column and last row has special values (each other reversed).
         */
        for (int i = 0; i < m; ++i) {
            Hdata[i][0] = Hdata[i][0] - hPowers[i];
            Hdata[m - 1][i] -= hPowers[m - i - 1];
        }

        /*
         * [1] states: "For 1/2 < h < 1 the bottom left element of the matrix should be (1 - 2*h^m +
         * (2h - 1)^m )/m!" Since 0 <= h < 1, then if h > 1/2 is sufficient to check:
         */
        if (Double.compare(h, 0.5) > 0) {
            Hdata[m - 1][0] += FastMath.pow(2 * h - 1, m);
        }

        /*
         * Aside from the first column and last row, the (i, j)-th element is 1/(i - j + 1)! if i -
         * j + 1 >= 0, else 0. 1's and 0's are already put, so only division with (i - j + 1)! is
         * needed in the elements that have 1's. There is no need to calculate (i - j + 1)! and then
         * divide - small steps avoid overflows. Note that i - j + 1 > 0 <=> i + 1 > j instead of
         * j'ing all the way to m. Also note that it is started at g = 2 because dividing by 1 isn't
         * really necessary.
         */
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                if (i - j + 1 > 0) {
                    for (int g = 2; g <= i - j + 1; ++g) {
                        Hdata[i][j] /= g;
                    }
                }
            }
        }
        return MatrixUtils.createRealMatrix(Hdata);
    }

    
    private void checkArray(double[] array) {
        if (array == null) {
            throw new NullArgumentException(LocalizedFormats.NULL_NOT_ALLOWED);
        }
        if (array.length < 2) {
            throw new InsufficientDataException(LocalizedFormats.INSUFFICIENT_OBSERVED_POINTS_IN_SAMPLE, array.length,
                                                2);
        }
    }

    
    public double ksSum(double t, double tolerance, int maxIterations) {
        if (t == 0.0) {
            return 0.0;
        }

        // TODO: for small t (say less than 1), the alternative expansion in part 3 of [1]
        // from class javadoc should be used.

        final double x = -2 * t * t;
        int sign = -1;
        long i = 1;
        double partialSum = 0.5d;
        double delta = 1;
        while (delta > tolerance && i < maxIterations) {
            delta = FastMath.exp(x * i * i);
            partialSum += sign * delta;
            sign *= -1;
            i++;
        }
        if (i == maxIterations) {
            throw new TooManyIterationsException(maxIterations);
        }
        return partialSum * 2;
    }

    
    private static long calculateIntegralD(double d, int n, int m, boolean strict) {
        final double tol = 1e-12;  // d-values within tol of one another are considered equal
        long nm = n * (long)m;
        long upperBound = (long)FastMath.ceil((d - tol) * nm);
        long lowerBound = (long)FastMath.floor((d + tol) * nm);
        if (strict && lowerBound == upperBound) {
            return upperBound + 1l;
        }
        else {
            return upperBound;
        }
    }

    
    public double exactP(double d, int n, int m, boolean strict) {
       return 1 - n(m, n, m, n, calculateIntegralD(d, m, n, strict), strict) /
               CombinatoricsUtils.binomialCoefficientDouble(n + m, m);
    }

    
    public double approximateP(double d, int n, int m) {
        final double dm = m;
        final double dn = n;
        return 1 - ksSum(d * FastMath.sqrt((dm * dn) / (dm + dn)),
                         KS_SUM_CAUCHY_CRITERION, MAXIMUM_PARTIAL_SUM_COUNT);
    }

    
    static void fillBooleanArrayRandomlyWithFixedNumberTrueValues(final boolean[] b, final int numberOfTrueValues, final RandomGenerator rng) {
        Arrays.fill(b, true);
        for (int k = numberOfTrueValues; k < b.length; k++) {
            final int r = rng.nextInt(k + 1);
            b[(b[r]) ? r : k] = false;
        }
    }

    
    public double monteCarloP(final double d, final int n, final int m, final boolean strict,
                              final int iterations) {
        return integralMonteCarloP(calculateIntegralD(d, n, m, strict), n, m, iterations);
    }

    
    private double integralMonteCarloP(final long d, final int n, final int m, final int iterations) {

        // ensure that nn is always the max of (n, m) to require fewer random numbers
        final int nn = FastMath.max(n, m);
        final int mm = FastMath.min(n, m);
        final int sum = nn + mm;

        int tail = 0;
        final boolean b[] = new boolean[sum];
        for (int i = 0; i < iterations; i++) {
            fillBooleanArrayRandomlyWithFixedNumberTrueValues(b, nn, rng);
            long curD = 0l;
            for(int j = 0; j < b.length; ++j) {
                if (b[j]) {
                    curD += mm;
                    if (curD >= d) {
                        tail++;
                        break;
                    }
                } else {
                    curD -= nn;
                    if (curD <= -d) {
                        tail++;
                        break;
                    }
                }
            }
        }
        return (double) tail / iterations;
    }

    
    private static void fixTies(double[] x, double[] y) {
       final double[] values = MathArrays.unique(MathArrays.concatenate(x,y));
       if (values.length == x.length + y.length) {
           return;  // There are no ties
       }

       // Find the smallest difference between values, or 1 if all values are the same
       double minDelta = 1;
       double prev = values[0];
       double delta = 1;
       for (int i = 1; i < values.length; i++) {
          delta = prev - values[i];
          if (delta < minDelta) {
              minDelta = delta;
          }
          prev = values[i];
       }
       minDelta /= 2;

       // Add jitter using a fixed seed (so same arguments always give same results),
       // low-initialization-overhead generator
       final RealDistribution dist =
               new UniformRealDistribution(new JDKRandomGenerator(100), -minDelta, minDelta);

       // It is theoretically possible that jitter does not break ties, so repeat
       // until all ties are gone.  Bound the loop and throw MIE if bound is exceeded.
       int ct = 0;
       boolean ties = true;
       do {
           jitter(x, dist);
           jitter(y, dist);
           ties = hasTies(x, y);
           ct++;
       } while (ties && ct < 1000);
       if (ties) {
           throw new MathInternalError(); // Should never happen
       }
    }

    
    private static boolean hasTies(double[] x, double[] y) {
        final HashSet<Double> values = new HashSet<Double>();
            for (int i = 0; i < x.length; i++) {
                if (!values.add(x[i])) {
                    return true;
                }
            }
            for (int i = 0; i < y.length; i++) {
                if (!values.add(y[i])) {
                    return true;
                }
            }
        return false;
    }

    
    private static void jitter(double[] data, RealDistribution dist) {
        for (int i = 0; i < data.length; i++) {
            data[i] += dist.sample();
        }
    }

    
    private static int c(int i, int j, int m, int n, long cmn, boolean strict) {
        if (strict) {
            return FastMath.abs(i*(long)n - j*(long)m) <= cmn ? 1 : 0;
        }
        return FastMath.abs(i*(long)n - j*(long)m) < cmn ? 1 : 0;
    }

    
    private static double n(int i, int j, int m, int n, long cnm, boolean strict) {
        /*
         * Unwind the recursive definition given in [4].
         * Compute n(1,1), n(1,2)...n(2,1), n(2,2)... up to n(i,j), one row at a time.
         * When n(i,*) are being computed, lag[] holds the values of n(i - 1, *).
         */
        final double[] lag = new double[n];
        double last = 0;
        for (int k = 0; k < n; k++) {
            lag[k] = c(0, k + 1, m, n, cnm, strict);
        }
        for (int k = 1; k <= i; k++) {
            last = c(k, 0, m, n, cnm, strict);
            for (int l = 1; l <= j; l++) {
                lag[l - 1] = c(k, l, m, n, cnm, strict) * (last + lag[l - 1]);
                last = lag[l - 1];
            }
        }
        return last;
    }
}
