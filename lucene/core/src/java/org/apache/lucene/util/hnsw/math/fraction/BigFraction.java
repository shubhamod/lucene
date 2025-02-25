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
package org.apache.lucene.util.hnsw.math.fraction;

import java.io.Serializable;
import java.math.BigDecimal;
import java.math.BigInteger;

import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.ArithmeticUtils;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class BigFraction
    extends Number
    implements FieldElement<BigFraction>, Comparable<BigFraction>, Serializable {

    
    public static final BigFraction TWO = new BigFraction(2);

    
    public static final BigFraction ONE = new BigFraction(1);

    
    public static final BigFraction ZERO = new BigFraction(0);

    
    public static final BigFraction MINUS_ONE = new BigFraction(-1);

    
    public static final BigFraction FOUR_FIFTHS = new BigFraction(4, 5);

    
    public static final BigFraction ONE_FIFTH = new BigFraction(1, 5);

    
    public static final BigFraction ONE_HALF = new BigFraction(1, 2);

    
    public static final BigFraction ONE_QUARTER = new BigFraction(1, 4);

    
    public static final BigFraction ONE_THIRD = new BigFraction(1, 3);

    
    public static final BigFraction THREE_FIFTHS = new BigFraction(3, 5);

    
    public static final BigFraction THREE_QUARTERS = new BigFraction(3, 4);

    
    public static final BigFraction TWO_FIFTHS = new BigFraction(2, 5);

    
    public static final BigFraction TWO_QUARTERS = new BigFraction(2, 4);

    
    public static final BigFraction TWO_THIRDS = new BigFraction(2, 3);

    
    private static final long serialVersionUID = -5630213147331578515L;

    
    private static final BigInteger ONE_HUNDRED = BigInteger.valueOf(100);

    
    private final BigInteger numerator;

    
    private final BigInteger denominator;

    
    public BigFraction(final BigInteger num) {
        this(num, BigInteger.ONE);
    }

    
    public BigFraction(BigInteger num, BigInteger den) {
        MathUtils.checkNotNull(num, LocalizedFormats.NUMERATOR);
        MathUtils.checkNotNull(den, LocalizedFormats.DENOMINATOR);
        if (den.signum() == 0) {
            throw new ZeroException(LocalizedFormats.ZERO_DENOMINATOR);
        }
        if (num.signum() == 0) {
            numerator   = BigInteger.ZERO;
            denominator = BigInteger.ONE;
        } else {

            // reduce numerator and denominator by greatest common denominator
            final BigInteger gcd = num.gcd(den);
            if (BigInteger.ONE.compareTo(gcd) < 0) {
                num = num.divide(gcd);
                den = den.divide(gcd);
            }

            // move sign to numerator
            if (den.signum() == -1) {
                num = num.negate();
                den = den.negate();
            }

            // store the values in the final fields
            numerator   = num;
            denominator = den;

        }
    }

    
    public BigFraction(final double value) throws MathIllegalArgumentException {
        if (Double.isNaN(value)) {
            throw new MathIllegalArgumentException(LocalizedFormats.NAN_VALUE_CONVERSION);
        }
        if (Double.isInfinite(value)) {
            throw new MathIllegalArgumentException(LocalizedFormats.INFINITE_VALUE_CONVERSION);
        }

        // compute m and k such that value = m * 2^k
        final long bits     = Double.doubleToLongBits(value);
        final long sign     = bits & 0x8000000000000000L;
        final long exponent = bits & 0x7ff0000000000000L;
        long m              = bits & 0x000fffffffffffffL;
        if (exponent != 0) {
            // this was a normalized number, add the implicit most significant bit
            m |= 0x0010000000000000L;
        }
        if (sign != 0) {
            m = -m;
        }
        int k = ((int) (exponent >> 52)) - 1075;
        while (((m & 0x001ffffffffffffeL) != 0) && ((m & 0x1) == 0)) {
            m >>= 1;
            ++k;
        }

        if (k < 0) {
            numerator   = BigInteger.valueOf(m);
            denominator = BigInteger.ZERO.flipBit(-k);
        } else {
            numerator   = BigInteger.valueOf(m).multiply(BigInteger.ZERO.flipBit(k));
            denominator = BigInteger.ONE;
        }

    }

    
    public BigFraction(final double value, final double epsilon,
                       final int maxIterations)
        throws FractionConversionException {
        this(value, epsilon, Integer.MAX_VALUE, maxIterations);
    }

    
    private BigFraction(final double value, final double epsilon,
                        final int maxDenominator, int maxIterations)
        throws FractionConversionException {
        long overflow = Integer.MAX_VALUE;
        double r0 = value;
        long a0 = (long) FastMath.floor(r0);

        if (FastMath.abs(a0) > overflow) {
            throw new FractionConversionException(value, a0, 1l);
        }

        // check for (almost) integer arguments, which should not go
        // to iterations.
        if (FastMath.abs(a0 - value) < epsilon) {
            numerator = BigInteger.valueOf(a0);
            denominator = BigInteger.ONE;
            return;
        }

        long p0 = 1;
        long q0 = 0;
        long p1 = a0;
        long q1 = 1;

        long p2 = 0;
        long q2 = 1;

        int n = 0;
        boolean stop = false;
        do {
            ++n;
            final double r1 = 1.0 / (r0 - a0);
            final long a1 = (long) FastMath.floor(r1);
            p2 = (a1 * p1) + p0;
            q2 = (a1 * q1) + q0;
            if ((p2 > overflow) || (q2 > overflow)) {
                // in maxDenominator mode, if the last fraction was very close to the actual value
                // q2 may overflow in the next iteration; in this case return the last one.
                if (epsilon == 0.0 && FastMath.abs(q1) < maxDenominator) {
                    break;
                }
                throw new FractionConversionException(value, p2, q2);
            }

            final double convergent = (double) p2 / (double) q2;
            if ((n < maxIterations) &&
                (FastMath.abs(convergent - value) > epsilon) &&
                (q2 < maxDenominator)) {
                p0 = p1;
                p1 = p2;
                q0 = q1;
                q1 = q2;
                a0 = a1;
                r0 = r1;
            } else {
                stop = true;
            }
        } while (!stop);

        if (n >= maxIterations) {
            throw new FractionConversionException(value, maxIterations);
        }

        if (q2 < maxDenominator) {
            numerator   = BigInteger.valueOf(p2);
            denominator = BigInteger.valueOf(q2);
        } else {
            numerator   = BigInteger.valueOf(p1);
            denominator = BigInteger.valueOf(q1);
        }
    }

    
    public BigFraction(final double value, final int maxDenominator)
        throws FractionConversionException {
        this(value, 0, maxDenominator, 100);
    }

    
    public BigFraction(final int num) {
        this(BigInteger.valueOf(num), BigInteger.ONE);
    }

    
    public BigFraction(final int num, final int den) {
        this(BigInteger.valueOf(num), BigInteger.valueOf(den));
    }

    
    public BigFraction(final long num) {
        this(BigInteger.valueOf(num), BigInteger.ONE);
    }

    
    public BigFraction(final long num, final long den) {
        this(BigInteger.valueOf(num), BigInteger.valueOf(den));
    }

    
    public static BigFraction getReducedFraction(final int numerator,
                                                 final int denominator) {
        if (numerator == 0) {
            return ZERO; // normalize zero.
        }

        return new BigFraction(numerator, denominator);
    }

    
    public BigFraction abs() {
        return (numerator.signum() == 1) ? this : negate();
    }

    
    public BigFraction add(final BigInteger bg) throws NullArgumentException {
        MathUtils.checkNotNull(bg);

        if (numerator.signum() == 0) {
            return new BigFraction(bg);
        }
        if (bg.signum() == 0) {
            return this;
        }

        return new BigFraction(numerator.add(denominator.multiply(bg)), denominator);
    }

    
    public BigFraction add(final int i) {
        return add(BigInteger.valueOf(i));
    }

    
    public BigFraction add(final long l) {
        return add(BigInteger.valueOf(l));
    }

    
    public BigFraction add(final BigFraction fraction) {
        if (fraction == null) {
            throw new NullArgumentException(LocalizedFormats.FRACTION);
        }
        if (fraction.numerator.signum() == 0) {
            return this;
        }
        if (numerator.signum() == 0) {
            return fraction;
        }

        BigInteger num = null;
        BigInteger den = null;

        if (denominator.equals(fraction.denominator)) {
            num = numerator.add(fraction.numerator);
            den = denominator;
        } else {
            num = (numerator.multiply(fraction.denominator)).add((fraction.numerator).multiply(denominator));
            den = denominator.multiply(fraction.denominator);
        }

        if (num.signum() == 0) {
            return ZERO;
        }

        return new BigFraction(num, den);

    }

    
    public BigDecimal bigDecimalValue() {
        return new BigDecimal(numerator).divide(new BigDecimal(denominator));
    }

    
    public BigDecimal bigDecimalValue(final int roundingMode) {
        return new BigDecimal(numerator).divide(new BigDecimal(denominator), roundingMode);
    }

    
    public BigDecimal bigDecimalValue(final int scale, final int roundingMode) {
        return new BigDecimal(numerator).divide(new BigDecimal(denominator), scale, roundingMode);
    }

    
    public int compareTo(final BigFraction object) {
        int lhsSigNum = numerator.signum();
        int rhsSigNum = object.numerator.signum();

        if (lhsSigNum != rhsSigNum) {
            return (lhsSigNum > rhsSigNum) ? 1 : -1;
        }
        if (lhsSigNum == 0) {
            return 0;
        }

        BigInteger nOd = numerator.multiply(object.denominator);
        BigInteger dOn = denominator.multiply(object.numerator);
        return nOd.compareTo(dOn);
    }

    
    public BigFraction divide(final BigInteger bg) {
        if (bg == null) {
            throw new NullArgumentException(LocalizedFormats.FRACTION);
        }
        if (bg.signum() == 0) {
            throw new MathArithmeticException(LocalizedFormats.ZERO_DENOMINATOR);
        }
        if (numerator.signum() == 0) {
            return ZERO;
        }
        return new BigFraction(numerator, denominator.multiply(bg));
    }

    
    public BigFraction divide(final int i) {
        return divide(BigInteger.valueOf(i));
    }

    
    public BigFraction divide(final long l) {
        return divide(BigInteger.valueOf(l));
    }

    
    public BigFraction divide(final BigFraction fraction) {
        if (fraction == null) {
            throw new NullArgumentException(LocalizedFormats.FRACTION);
        }
        if (fraction.numerator.signum() == 0) {
            throw new MathArithmeticException(LocalizedFormats.ZERO_DENOMINATOR);
        }
        if (numerator.signum() == 0) {
            return ZERO;
        }

        return multiply(fraction.reciprocal());
    }

    
    @Override
    public double doubleValue() {
        double result = numerator.doubleValue() / denominator.doubleValue();
        if (Double.isNaN(result)) {
            // Numerator and/or denominator must be out of range:
            // Calculate how far to shift them to put them in range.
            int shift = FastMath.max(numerator.bitLength(),
                                     denominator.bitLength()) - FastMath.getExponent(Double.MAX_VALUE);
            result = numerator.shiftRight(shift).doubleValue() /
                denominator.shiftRight(shift).doubleValue();
        }
        return result;
    }

    
    @Override
    public boolean equals(final Object other) {
        boolean ret = false;

        if (this == other) {
            ret = true;
        } else if (other instanceof BigFraction) {
            BigFraction rhs = ((BigFraction) other).reduce();
            BigFraction thisOne = this.reduce();
            ret = thisOne.numerator.equals(rhs.numerator) && thisOne.denominator.equals(rhs.denominator);
        }

        return ret;
    }

    
    @Override
    public float floatValue() {
        float result = numerator.floatValue() / denominator.floatValue();
        if (Double.isNaN(result)) {
            // Numerator and/or denominator must be out of range:
            // Calculate how far to shift them to put them in range.
            int shift = FastMath.max(numerator.bitLength(),
                                     denominator.bitLength()) - FastMath.getExponent(Float.MAX_VALUE);
            result = numerator.shiftRight(shift).floatValue() /
                denominator.shiftRight(shift).floatValue();
        }
        return result;
    }

    
    public BigInteger getDenominator() {
        return denominator;
    }

    
    public int getDenominatorAsInt() {
        return denominator.intValue();
    }

    
    public long getDenominatorAsLong() {
        return denominator.longValue();
    }

    
    public BigInteger getNumerator() {
        return numerator;
    }

    
    public int getNumeratorAsInt() {
        return numerator.intValue();
    }

    
    public long getNumeratorAsLong() {
        return numerator.longValue();
    }

    
    @Override
    public int hashCode() {
        return 37 * (37 * 17 + numerator.hashCode()) + denominator.hashCode();
    }

    
    @Override
    public int intValue() {
        return numerator.divide(denominator).intValue();
    }

    
    @Override
    public long longValue() {
        return numerator.divide(denominator).longValue();
    }

    
    public BigFraction multiply(final BigInteger bg) {
        if (bg == null) {
            throw new NullArgumentException();
        }
        if (numerator.signum() == 0 || bg.signum() == 0) {
            return ZERO;
        }
        return new BigFraction(bg.multiply(numerator), denominator);
    }

    
    public BigFraction multiply(final int i) {
        if (i == 0 || numerator.signum() == 0) {
            return ZERO;
        }

        return multiply(BigInteger.valueOf(i));
    }

    
    public BigFraction multiply(final long l) {
        if (l == 0 || numerator.signum() == 0) {
            return ZERO;
        }

        return multiply(BigInteger.valueOf(l));
    }

    
    public BigFraction multiply(final BigFraction fraction) {
        if (fraction == null) {
            throw new NullArgumentException(LocalizedFormats.FRACTION);
        }
        if (numerator.signum() == 0 ||
            fraction.numerator.signum() == 0) {
            return ZERO;
        }
        return new BigFraction(numerator.multiply(fraction.numerator),
                               denominator.multiply(fraction.denominator));
    }

    
    public BigFraction negate() {
        return new BigFraction(numerator.negate(), denominator);
    }

    
    public double percentageValue() {
        return multiply(ONE_HUNDRED).doubleValue();
    }

    
    public BigFraction pow(final int exponent) {
        if (exponent == 0) {
            return ONE;
        }
        if (numerator.signum() == 0) {
            return this;
        }

        if (exponent < 0) {
            return new BigFraction(denominator.pow(-exponent), numerator.pow(-exponent));
        }
        return new BigFraction(numerator.pow(exponent), denominator.pow(exponent));
    }

    
    public BigFraction pow(final long exponent) {
        if (exponent == 0) {
            return ONE;
        }
        if (numerator.signum() == 0) {
            return this;
        }

        if (exponent < 0) {
            return new BigFraction(ArithmeticUtils.pow(denominator, -exponent),
                                   ArithmeticUtils.pow(numerator,   -exponent));
        }
        return new BigFraction(ArithmeticUtils.pow(numerator,   exponent),
                               ArithmeticUtils.pow(denominator, exponent));
    }

    
    public BigFraction pow(final BigInteger exponent) {
        if (exponent.signum() == 0) {
            return ONE;
        }
        if (numerator.signum() == 0) {
            return this;
        }

        if (exponent.signum() == -1) {
            final BigInteger eNeg = exponent.negate();
            return new BigFraction(ArithmeticUtils.pow(denominator, eNeg),
                                   ArithmeticUtils.pow(numerator,   eNeg));
        }
        return new BigFraction(ArithmeticUtils.pow(numerator,   exponent),
                               ArithmeticUtils.pow(denominator, exponent));
    }

    
    public double pow(final double exponent) {
        return FastMath.pow(numerator.doubleValue(),   exponent) /
               FastMath.pow(denominator.doubleValue(), exponent);
    }

    
    public BigFraction reciprocal() {
        return new BigFraction(denominator, numerator);
    }

    
    public BigFraction reduce() {
        final BigInteger gcd = numerator.gcd(denominator);

        if (BigInteger.ONE.compareTo(gcd) < 0) {
            return new BigFraction(numerator.divide(gcd), denominator.divide(gcd));
        } else {
            return this;
        }
    }

    
    public BigFraction subtract(final BigInteger bg) {
        if (bg == null) {
            throw new NullArgumentException();
        }
        if (bg.signum() == 0) {
            return this;
        }
        if (numerator.signum() == 0) {
            return new BigFraction(bg.negate());
        }

        return new BigFraction(numerator.subtract(denominator.multiply(bg)), denominator);
    }

    
    public BigFraction subtract(final int i) {
        return subtract(BigInteger.valueOf(i));
    }

    
    public BigFraction subtract(final long l) {
        return subtract(BigInteger.valueOf(l));
    }

    
    public BigFraction subtract(final BigFraction fraction) {
        if (fraction == null) {
            throw new NullArgumentException(LocalizedFormats.FRACTION);
        }
        if (fraction.numerator.signum() == 0) {
            return this;
        }
        if (numerator.signum() == 0) {
            return fraction.negate();
        }

        BigInteger num = null;
        BigInteger den = null;
        if (denominator.equals(fraction.denominator)) {
            num = numerator.subtract(fraction.numerator);
            den = denominator;
        } else {
            num = (numerator.multiply(fraction.denominator)).subtract((fraction.numerator).multiply(denominator));
            den = denominator.multiply(fraction.denominator);
        }
        return new BigFraction(num, den);

    }

    
    @Override
    public String toString() {
        String str = null;
        if (BigInteger.ONE.equals(denominator)) {
            str = numerator.toString();
        } else if (BigInteger.ZERO.equals(numerator)) {
            str = "0";
        } else {
            str = numerator + " / " + denominator;
        }
        return str;
    }

    
    public BigFractionField getField() {
        return BigFractionField.getInstance();
    }

}
