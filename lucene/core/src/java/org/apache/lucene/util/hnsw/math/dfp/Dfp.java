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

package org.apache.lucene.util.hnsw.math.dfp;

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class Dfp implements RealFieldElement<Dfp> {

    
    public static final int RADIX = 10000;

    
    public static final int MIN_EXP = -32767;

    
    public static final int MAX_EXP =  32768;

    
    public static final int ERR_SCALE = 32760;

    
    public static final byte FINITE = 0;

    
    public static final byte INFINITE = 1;

    
    public static final byte SNAN = 2;

    
    public static final byte QNAN = 3;

    
    private static final String NAN_STRING = "NaN";

    
    private static final String POS_INFINITY_STRING = "Infinity";

    
    private static final String NEG_INFINITY_STRING = "-Infinity";

    
    private static final String ADD_TRAP = "add";

    
    private static final String MULTIPLY_TRAP = "multiply";

    
    private static final String DIVIDE_TRAP = "divide";

    
    private static final String SQRT_TRAP = "sqrt";

    
    private static final String ALIGN_TRAP = "align";

    
    private static final String TRUNC_TRAP = "trunc";

    
    private static final String NEXT_AFTER_TRAP = "nextAfter";

    
    private static final String LESS_THAN_TRAP = "lessThan";

    
    private static final String GREATER_THAN_TRAP = "greaterThan";

    
    private static final String NEW_INSTANCE_TRAP = "newInstance";

    
    protected int[] mant;

    
    protected byte sign;

    
    protected int exp;

    
    protected byte nans;

    
    private final DfpField field;

    
    protected Dfp(final DfpField field) {
        mant = new int[field.getRadixDigits()];
        sign = 1;
        exp = 0;
        nans = FINITE;
        this.field = field;
    }

    
    protected Dfp(final DfpField field, byte x) {
        this(field, (long) x);
    }

    
    protected Dfp(final DfpField field, int x) {
        this(field, (long) x);
    }

    
    protected Dfp(final DfpField field, long x) {

        // initialize as if 0
        mant = new int[field.getRadixDigits()];
        nans = FINITE;
        this.field = field;

        boolean isLongMin = false;
        if (x == Long.MIN_VALUE) {
            // special case for Long.MIN_VALUE (-9223372036854775808)
            // we must shift it before taking its absolute value
            isLongMin = true;
            ++x;
        }

        // set the sign
        if (x < 0) {
            sign = -1;
            x = -x;
        } else {
            sign = 1;
        }

        exp = 0;
        while (x != 0) {
            System.arraycopy(mant, mant.length - exp, mant, mant.length - 1 - exp, exp);
            mant[mant.length - 1] = (int) (x % RADIX);
            x /= RADIX;
            exp++;
        }

        if (isLongMin) {
            // remove the shift added for Long.MIN_VALUE
            // we know in this case that fixing the last digit is sufficient
            for (int i = 0; i < mant.length - 1; i++) {
                if (mant[i] != 0) {
                    mant[i]++;
                    break;
                }
            }
        }
    }

    
    protected Dfp(final DfpField field, double x) {

        // initialize as if 0
        mant = new int[field.getRadixDigits()];
        sign = 1;
        exp = 0;
        nans = FINITE;
        this.field = field;

        long bits = Double.doubleToLongBits(x);
        long mantissa = bits & 0x000fffffffffffffL;
        int exponent = (int) ((bits & 0x7ff0000000000000L) >> 52) - 1023;

        if (exponent == -1023) {
            // Zero or sub-normal
            if (x == 0) {
                // make sure 0 has the right sign
                if ((bits & 0x8000000000000000L) != 0) {
                    sign = -1;
                }
                return;
            }

            exponent++;

            // Normalize the subnormal number
            while ( (mantissa & 0x0010000000000000L) == 0) {
                exponent--;
                mantissa <<= 1;
            }
            mantissa &= 0x000fffffffffffffL;
        }

        if (exponent == 1024) {
            // infinity or NAN
            if (x != x) {
                sign = (byte) 1;
                nans = QNAN;
            } else if (x < 0) {
                sign = (byte) -1;
                nans = INFINITE;
            } else {
                sign = (byte) 1;
                nans = INFINITE;
            }
            return;
        }

        Dfp xdfp = new Dfp(field, mantissa);
        xdfp = xdfp.divide(new Dfp(field, 4503599627370496l)).add(field.getOne());  // Divide by 2^52, then add one
        xdfp = xdfp.multiply(DfpMath.pow(field.getTwo(), exponent));

        if ((bits & 0x8000000000000000L) != 0) {
            xdfp = xdfp.negate();
        }

        System.arraycopy(xdfp.mant, 0, mant, 0, mant.length);
        sign = xdfp.sign;
        exp  = xdfp.exp;
        nans = xdfp.nans;

    }

    
    public Dfp(final Dfp d) {
        mant  = d.mant.clone();
        sign  = d.sign;
        exp   = d.exp;
        nans  = d.nans;
        field = d.field;
    }

    
    protected Dfp(final DfpField field, final String s) {

        // initialize as if 0
        mant = new int[field.getRadixDigits()];
        sign = 1;
        exp = 0;
        nans = FINITE;
        this.field = field;

        boolean decimalFound = false;
        final int rsize = 4;   // size of radix in decimal digits
        final int offset = 4;  // Starting offset into Striped
        final char[] striped = new char[getRadixDigits() * rsize + offset * 2];

        // Check some special cases
        if (s.equals(POS_INFINITY_STRING)) {
            sign = (byte) 1;
            nans = INFINITE;
            return;
        }

        if (s.equals(NEG_INFINITY_STRING)) {
            sign = (byte) -1;
            nans = INFINITE;
            return;
        }

        if (s.equals(NAN_STRING)) {
            sign = (byte) 1;
            nans = QNAN;
            return;
        }

        // Check for scientific notation
        int p = s.indexOf("e");
        if (p == -1) { // try upper case?
            p = s.indexOf("E");
        }

        final String fpdecimal;
        int sciexp = 0;
        if (p != -1) {
            // scientific notation
            fpdecimal = s.substring(0, p);
            String fpexp = s.substring(p+1);
            boolean negative = false;

            for (int i=0; i<fpexp.length(); i++)
            {
                if (fpexp.charAt(i) == '-')
                {
                    negative = true;
                    continue;
                }
                if (fpexp.charAt(i) >= '0' && fpexp.charAt(i) <= '9') {
                    sciexp = sciexp * 10 + fpexp.charAt(i) - '0';
                }
            }

            if (negative) {
                sciexp = -sciexp;
            }
        } else {
            // normal case
            fpdecimal = s;
        }

        // If there is a minus sign in the number then it is negative
        if (fpdecimal.indexOf("-") !=  -1) {
            sign = -1;
        }

        // First off, find all of the leading zeros, trailing zeros, and significant digits
        p = 0;

        // Move p to first significant digit
        int decimalPos = 0;
        for (;;) {
            if (fpdecimal.charAt(p) >= '1' && fpdecimal.charAt(p) <= '9') {
                break;
            }

            if (decimalFound && fpdecimal.charAt(p) == '0') {
                decimalPos--;
            }

            if (fpdecimal.charAt(p) == '.') {
                decimalFound = true;
            }

            p++;

            if (p == fpdecimal.length()) {
                break;
            }
        }

        // Copy the string onto Stripped
        int q = offset;
        striped[0] = '0';
        striped[1] = '0';
        striped[2] = '0';
        striped[3] = '0';
        int significantDigits=0;
        for(;;) {
            if (p == (fpdecimal.length())) {
                break;
            }

            // Don't want to run pass the end of the array
            if (q == mant.length*rsize+offset+1) {
                break;
            }

            if (fpdecimal.charAt(p) == '.') {
                decimalFound = true;
                decimalPos = significantDigits;
                p++;
                continue;
            }

            if (fpdecimal.charAt(p) < '0' || fpdecimal.charAt(p) > '9') {
                p++;
                continue;
            }

            striped[q] = fpdecimal.charAt(p);
            q++;
            p++;
            significantDigits++;
        }


        // If the decimal point has been found then get rid of trailing zeros.
        if (decimalFound && q != offset) {
            for (;;) {
                q--;
                if (q == offset) {
                    break;
                }
                if (striped[q] == '0') {
                    significantDigits--;
                } else {
                    break;
                }
            }
        }

        // special case of numbers like "0.00000"
        if (decimalFound && significantDigits == 0) {
            decimalPos = 0;
        }

        // Implicit decimal point at end of number if not present
        if (!decimalFound) {
            decimalPos = q-offset;
        }

        // Find the number of significant trailing zeros
        q = offset;  // set q to point to first sig digit
        p = significantDigits-1+offset;

        while (p > q) {
            if (striped[p] != '0') {
                break;
            }
            p--;
        }

        // Make sure the decimal is on a mod 10000 boundary
        int i = ((rsize * 100) - decimalPos - sciexp % rsize) % rsize;
        q -= i;
        decimalPos += i;

        // Make the mantissa length right by adding zeros at the end if necessary
        while ((p - q) < (mant.length * rsize)) {
            for (i = 0; i < rsize; i++) {
                striped[++p] = '0';
            }
        }

        // Ok, now we know how many trailing zeros there are,
        // and where the least significant digit is
        for (i = mant.length - 1; i >= 0; i--) {
            mant[i] = (striped[q]   - '0') * 1000 +
                      (striped[q+1] - '0') * 100  +
                      (striped[q+2] - '0') * 10   +
                      (striped[q+3] - '0');
            q += 4;
        }


        exp = (decimalPos+sciexp) / rsize;

        if (q < striped.length) {
            // Is there possible another digit?
            round((striped[q] - '0')*1000);
        }

    }

    
    protected Dfp(final DfpField field, final byte sign, final byte nans) {
        this.field = field;
        this.mant    = new int[field.getRadixDigits()];
        this.sign    = sign;
        this.exp     = 0;
        this.nans    = nans;
    }

    
    public Dfp newInstance() {
        return new Dfp(getField());
    }

    
    public Dfp newInstance(final byte x) {
        return new Dfp(getField(), x);
    }

    
    public Dfp newInstance(final int x) {
        return new Dfp(getField(), x);
    }

    
    public Dfp newInstance(final long x) {
        return new Dfp(getField(), x);
    }

    
    public Dfp newInstance(final double x) {
        return new Dfp(getField(), x);
    }

    
    public Dfp newInstance(final Dfp d) {

        // make sure we don't mix number with different precision
        if (field.getRadixDigits() != d.field.getRadixDigits()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            final Dfp result = newInstance(getZero());
            result.nans = QNAN;
            return dotrap(DfpField.FLAG_INVALID, NEW_INSTANCE_TRAP, d, result);
        }

        return new Dfp(d);

    }

    
    public Dfp newInstance(final String s) {
        return new Dfp(field, s);
    }

    
    public Dfp newInstance(final byte sig, final byte code) {
        return field.newDfp(sig, code);
    }

    
    public DfpField getField() {
        return field;
    }

    
    public int getRadixDigits() {
        return field.getRadixDigits();
    }

    
    public Dfp getZero() {
        return field.getZero();
    }

    
    public Dfp getOne() {
        return field.getOne();
    }

    
    public Dfp getTwo() {
        return field.getTwo();
    }

    
    protected void shiftLeft() {
        for (int i = mant.length - 1; i > 0; i--) {
            mant[i] = mant[i-1];
        }
        mant[0] = 0;
        exp--;
    }

    /* Note that shiftRight() does not call round() as that round() itself
     uses shiftRight() */
    
    protected void shiftRight() {
        for (int i = 0; i < mant.length - 1; i++) {
            mant[i] = mant[i+1];
        }
        mant[mant.length - 1] = 0;
        exp++;
    }

    
    protected int align(int e) {
        int lostdigit = 0;
        boolean inexact = false;

        int diff = exp - e;

        int adiff = diff;
        if (adiff < 0) {
            adiff = -adiff;
        }

        if (diff == 0) {
            return 0;
        }

        if (adiff > (mant.length + 1)) {
            // Special case
            Arrays.fill(mant, 0);
            exp = e;

            field.setIEEEFlagsBits(DfpField.FLAG_INEXACT);
            dotrap(DfpField.FLAG_INEXACT, ALIGN_TRAP, this, this);

            return 0;
        }

        for (int i = 0; i < adiff; i++) {
            if (diff < 0) {
                /* Keep track of loss -- only signal inexact after losing 2 digits.
                 * the first lost digit is returned to add() and may be incorporated
                 * into the result.
                 */
                if (lostdigit != 0) {
                    inexact = true;
                }

                lostdigit = mant[0];

                shiftRight();
            } else {
                shiftLeft();
            }
        }

        if (inexact) {
            field.setIEEEFlagsBits(DfpField.FLAG_INEXACT);
            dotrap(DfpField.FLAG_INEXACT, ALIGN_TRAP, this, this);
        }

        return lostdigit;

    }

    
    public boolean lessThan(final Dfp x) {

        // make sure we don't mix number with different precision
        if (field.getRadixDigits() != x.field.getRadixDigits()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            final Dfp result = newInstance(getZero());
            result.nans = QNAN;
            dotrap(DfpField.FLAG_INVALID, LESS_THAN_TRAP, x, result);
            return false;
        }

        /* if a nan is involved, signal invalid and return false */
        if (isNaN() || x.isNaN()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            dotrap(DfpField.FLAG_INVALID, LESS_THAN_TRAP, x, newInstance(getZero()));
            return false;
        }

        return compare(this, x) < 0;
    }

    
    public boolean greaterThan(final Dfp x) {

        // make sure we don't mix number with different precision
        if (field.getRadixDigits() != x.field.getRadixDigits()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            final Dfp result = newInstance(getZero());
            result.nans = QNAN;
            dotrap(DfpField.FLAG_INVALID, GREATER_THAN_TRAP, x, result);
            return false;
        }

        /* if a nan is involved, signal invalid and return false */
        if (isNaN() || x.isNaN()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            dotrap(DfpField.FLAG_INVALID, GREATER_THAN_TRAP, x, newInstance(getZero()));
            return false;
        }

        return compare(this, x) > 0;
    }

    
    public boolean negativeOrNull() {

        if (isNaN()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            dotrap(DfpField.FLAG_INVALID, LESS_THAN_TRAP, this, newInstance(getZero()));
            return false;
        }

        return (sign < 0) || ((mant[mant.length - 1] == 0) && !isInfinite());

    }

    
    public boolean strictlyNegative() {

        if (isNaN()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            dotrap(DfpField.FLAG_INVALID, LESS_THAN_TRAP, this, newInstance(getZero()));
            return false;
        }

        return (sign < 0) && ((mant[mant.length - 1] != 0) || isInfinite());

    }

    
    public boolean positiveOrNull() {

        if (isNaN()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            dotrap(DfpField.FLAG_INVALID, LESS_THAN_TRAP, this, newInstance(getZero()));
            return false;
        }

        return (sign > 0) || ((mant[mant.length - 1] == 0) && !isInfinite());

    }

    
    public boolean strictlyPositive() {

        if (isNaN()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            dotrap(DfpField.FLAG_INVALID, LESS_THAN_TRAP, this, newInstance(getZero()));
            return false;
        }

        return (sign > 0) && ((mant[mant.length - 1] != 0) || isInfinite());

    }

    
    public Dfp abs() {
        Dfp result = newInstance(this);
        result.sign = 1;
        return result;
    }

    
    public boolean isInfinite() {
        return nans == INFINITE;
    }

    
    public boolean isNaN() {
        return (nans == QNAN) || (nans == SNAN);
    }

    
    public boolean isZero() {

        if (isNaN()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            dotrap(DfpField.FLAG_INVALID, LESS_THAN_TRAP, this, newInstance(getZero()));
            return false;
        }

        return (mant[mant.length - 1] == 0) && !isInfinite();

    }

    
    @Override
    public boolean equals(final Object other) {

        if (other instanceof Dfp) {
            final Dfp x = (Dfp) other;
            if (isNaN() || x.isNaN() || field.getRadixDigits() != x.field.getRadixDigits()) {
                return false;
            }

            return compare(this, x) == 0;
        }

        return false;

    }

    
    @Override
    public int hashCode() {
        return 17 + (isZero() ? 0 : (sign << 8)) + (nans << 16) + exp + Arrays.hashCode(mant);
    }

    
    public boolean unequal(final Dfp x) {
        if (isNaN() || x.isNaN() || field.getRadixDigits() != x.field.getRadixDigits()) {
            return false;
        }

        return greaterThan(x) || lessThan(x);
    }

    
    private static int compare(final Dfp a, final Dfp b) {
        // Ignore the sign of zero
        if (a.mant[a.mant.length - 1] == 0 && b.mant[b.mant.length - 1] == 0 &&
            a.nans == FINITE && b.nans == FINITE) {
            return 0;
        }

        if (a.sign != b.sign) {
            if (a.sign == -1) {
                return -1;
            } else {
                return 1;
            }
        }

        // deal with the infinities
        if (a.nans == INFINITE && b.nans == FINITE) {
            return a.sign;
        }

        if (a.nans == FINITE && b.nans == INFINITE) {
            return -b.sign;
        }

        if (a.nans == INFINITE && b.nans == INFINITE) {
            return 0;
        }

        // Handle special case when a or b is zero, by ignoring the exponents
        if (b.mant[b.mant.length-1] != 0 && a.mant[b.mant.length-1] != 0) {
            if (a.exp < b.exp) {
                return -a.sign;
            }

            if (a.exp > b.exp) {
                return a.sign;
            }
        }

        // compare the mantissas
        for (int i = a.mant.length - 1; i >= 0; i--) {
            if (a.mant[i] > b.mant[i]) {
                return a.sign;
            }

            if (a.mant[i] < b.mant[i]) {
                return -a.sign;
            }
        }

        return 0;

    }

    
    public Dfp rint() {
        return trunc(DfpField.RoundingMode.ROUND_HALF_EVEN);
    }

    
    public Dfp floor() {
        return trunc(DfpField.RoundingMode.ROUND_FLOOR);
    }

    
    public Dfp ceil() {
        return trunc(DfpField.RoundingMode.ROUND_CEIL);
    }

    
    public Dfp remainder(final Dfp d) {

        final Dfp result = this.subtract(this.divide(d).rint().multiply(d));

        // IEEE 854-1987 says that if the result is zero, then it carries the sign of this
        if (result.mant[mant.length-1] == 0) {
            result.sign = sign;
        }

        return result;

    }

    
    protected Dfp trunc(final DfpField.RoundingMode rmode) {
        boolean changed = false;

        if (isNaN()) {
            return newInstance(this);
        }

        if (nans == INFINITE) {
            return newInstance(this);
        }

        if (mant[mant.length-1] == 0) {
            // a is zero
            return newInstance(this);
        }

        /* If the exponent is less than zero then we can certainly
         * return zero */
        if (exp < 0) {
            field.setIEEEFlagsBits(DfpField.FLAG_INEXACT);
            Dfp result = newInstance(getZero());
            result = dotrap(DfpField.FLAG_INEXACT, TRUNC_TRAP, this, result);
            return result;
        }

        /* If the exponent is greater than or equal to digits, then it
         * must already be an integer since there is no precision left
         * for any fractional part */

        if (exp >= mant.length) {
            return newInstance(this);
        }

        /* General case:  create another dfp, result, that contains the
         * a with the fractional part lopped off.  */

        Dfp result = newInstance(this);
        for (int i = 0; i < mant.length-result.exp; i++) {
            changed |= result.mant[i] != 0;
            result.mant[i] = 0;
        }

        if (changed) {
            switch (rmode) {
                case ROUND_FLOOR:
                    if (result.sign == -1) {
                        // then we must increment the mantissa by one
                        result = result.add(newInstance(-1));
                    }
                    break;

                case ROUND_CEIL:
                    if (result.sign == 1) {
                        // then we must increment the mantissa by one
                        result = result.add(getOne());
                    }
                    break;

                case ROUND_HALF_EVEN:
                default:
                    final Dfp half = newInstance("0.5");
                    Dfp a = subtract(result);  // difference between this and result
                    a.sign = 1;            // force positive (take abs)
                    if (a.greaterThan(half)) {
                        a = newInstance(getOne());
                        a.sign = sign;
                        result = result.add(a);
                    }

                    
                    if (a.equals(half) && result.exp > 0 && (result.mant[mant.length-result.exp]&1) != 0) {
                        a = newInstance(getOne());
                        a.sign = sign;
                        result = result.add(a);
                    }
                    break;
            }

            field.setIEEEFlagsBits(DfpField.FLAG_INEXACT);  // signal inexact
            result = dotrap(DfpField.FLAG_INEXACT, TRUNC_TRAP, this, result);
            return result;
        }

        return result;
    }

    
    public int intValue() {
        Dfp rounded;
        int result = 0;

        rounded = rint();

        if (rounded.greaterThan(newInstance(2147483647))) {
            return 2147483647;
        }

        if (rounded.lessThan(newInstance(-2147483648))) {
            return -2147483648;
        }

        for (int i = mant.length - 1; i >= mant.length - rounded.exp; i--) {
            result = result * RADIX + rounded.mant[i];
        }

        if (rounded.sign == -1) {
            result = -result;
        }

        return result;
    }

    
    public int log10K() {
        return exp - 1;
    }

    
    public Dfp power10K(final int e) {
        Dfp d = newInstance(getOne());
        d.exp = e + 1;
        return d;
    }

    
    public int intLog10()  {
        if (mant[mant.length-1] > 1000) {
            return exp * 4 - 1;
        }
        if (mant[mant.length-1] > 100) {
            return exp * 4 - 2;
        }
        if (mant[mant.length-1] > 10) {
            return exp * 4 - 3;
        }
        return exp * 4 - 4;
    }

    
    public Dfp power10(final int e) {
        Dfp d = newInstance(getOne());

        if (e >= 0) {
            d.exp = e / 4 + 1;
        } else {
            d.exp = (e + 1) / 4;
        }

        switch ((e % 4 + 4) % 4) {
            case 0:
                break;
            case 1:
                d = d.multiply(10);
                break;
            case 2:
                d = d.multiply(100);
                break;
            default:
                d = d.multiply(1000);
        }

        return d;
    }

    
    protected int complement(int extra) {

        extra = RADIX-extra;
        for (int i = 0; i < mant.length; i++) {
            mant[i] = RADIX-mant[i]-1;
        }

        int rh = extra / RADIX;
        extra -= rh * RADIX;
        for (int i = 0; i < mant.length; i++) {
            final int r = mant[i] + rh;
            rh = r / RADIX;
            mant[i] = r - rh * RADIX;
        }

        return extra;
    }

    
    public Dfp add(final Dfp x) {

        // make sure we don't mix number with different precision
        if (field.getRadixDigits() != x.field.getRadixDigits()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            final Dfp result = newInstance(getZero());
            result.nans = QNAN;
            return dotrap(DfpField.FLAG_INVALID, ADD_TRAP, x, result);
        }

        /* handle special cases */
        if (nans != FINITE || x.nans != FINITE) {
            if (isNaN()) {
                return this;
            }

            if (x.isNaN()) {
                return x;
            }

            if (nans == INFINITE && x.nans == FINITE) {
                return this;
            }

            if (x.nans == INFINITE && nans == FINITE) {
                return x;
            }

            if (x.nans == INFINITE && nans == INFINITE && sign == x.sign) {
                return x;
            }

            if (x.nans == INFINITE && nans == INFINITE && sign != x.sign) {
                field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
                Dfp result = newInstance(getZero());
                result.nans = QNAN;
                result = dotrap(DfpField.FLAG_INVALID, ADD_TRAP, x, result);
                return result;
            }
        }

        /* copy this and the arg */
        Dfp a = newInstance(this);
        Dfp b = newInstance(x);

        /* initialize the result object */
        Dfp result = newInstance(getZero());

        /* Make all numbers positive, but remember their sign */
        final byte asign = a.sign;
        final byte bsign = b.sign;

        a.sign = 1;
        b.sign = 1;

        /* The result will be signed like the arg with greatest magnitude */
        byte rsign = bsign;
        if (compare(a, b) > 0) {
            rsign = asign;
        }

        /* Handle special case when a or b is zero, by setting the exponent
       of the zero number equal to the other one.  This avoids an alignment
       which would cause catastropic loss of precision */
        if (b.mant[mant.length-1] == 0) {
            b.exp = a.exp;
        }

        if (a.mant[mant.length-1] == 0) {
            a.exp = b.exp;
        }

        /* align number with the smaller exponent */
        int aextradigit = 0;
        int bextradigit = 0;
        if (a.exp < b.exp) {
            aextradigit = a.align(b.exp);
        } else {
            bextradigit = b.align(a.exp);
        }

        /* complement the smaller of the two if the signs are different */
        if (asign != bsign) {
            if (asign == rsign) {
                bextradigit = b.complement(bextradigit);
            } else {
                aextradigit = a.complement(aextradigit);
            }
        }

        /* add the mantissas */
        int rh = 0; /* acts as a carry */
        for (int i = 0; i < mant.length; i++) {
            final int r = a.mant[i]+b.mant[i]+rh;
            rh = r / RADIX;
            result.mant[i] = r - rh * RADIX;
        }
        result.exp = a.exp;
        result.sign = rsign;

        /* handle overflow -- note, when asign!=bsign an overflow is
         * normal and should be ignored.  */

        if (rh != 0 && (asign == bsign)) {
            final int lostdigit = result.mant[0];
            result.shiftRight();
            result.mant[mant.length-1] = rh;
            final int excp = result.round(lostdigit);
            if (excp != 0) {
                result = dotrap(excp, ADD_TRAP, x, result);
            }
        }

        /* normalize the result */
        for (int i = 0; i < mant.length; i++) {
            if (result.mant[mant.length-1] != 0) {
                break;
            }
            result.shiftLeft();
            if (i == 0) {
                result.mant[0] = aextradigit+bextradigit;
                aextradigit = 0;
                bextradigit = 0;
            }
        }

        /* result is zero if after normalization the most sig. digit is zero */
        if (result.mant[mant.length-1] == 0) {
            result.exp = 0;

            if (asign != bsign) {
                // Unless adding 2 negative zeros, sign is positive
                result.sign = 1;  // Per IEEE 854-1987 Section 6.3
            }
        }

        /* Call round to test for over/under flows */
        final int excp = result.round(aextradigit + bextradigit);
        if (excp != 0) {
            result = dotrap(excp, ADD_TRAP, x, result);
        }

        return result;
    }

    
    public Dfp negate() {
        Dfp result = newInstance(this);
        result.sign = (byte) - result.sign;
        return result;
    }

    
    public Dfp subtract(final Dfp x) {
        return add(x.negate());
    }

    
    protected int round(int n) {
        boolean inc = false;
        switch (field.getRoundingMode()) {
            case ROUND_DOWN:
                inc = false;
                break;

            case ROUND_UP:
                inc = n != 0;       // round up if n!=0
                break;

            case ROUND_HALF_UP:
                inc = n >= 5000;  // round half up
                break;

            case ROUND_HALF_DOWN:
                inc = n > 5000;  // round half down
                break;

            case ROUND_HALF_EVEN:
                inc = n > 5000 || (n == 5000 && (mant[0] & 1) == 1);  // round half-even
                break;

            case ROUND_HALF_ODD:
                inc = n > 5000 || (n == 5000 && (mant[0] & 1) == 0);  // round half-odd
                break;

            case ROUND_CEIL:
                inc = sign == 1 && n != 0;  // round ceil
                break;

            case ROUND_FLOOR:
            default:
                inc = sign == -1 && n != 0;  // round floor
                break;
        }

        if (inc) {
            // increment if necessary
            int rh = 1;
            for (int i = 0; i < mant.length; i++) {
                final int r = mant[i] + rh;
                rh = r / RADIX;
                mant[i] = r - rh * RADIX;
            }

            if (rh != 0) {
                shiftRight();
                mant[mant.length-1] = rh;
            }
        }

        // check for exceptional cases and raise signals if necessary
        if (exp < MIN_EXP) {
            // Gradual Underflow
            field.setIEEEFlagsBits(DfpField.FLAG_UNDERFLOW);
            return DfpField.FLAG_UNDERFLOW;
        }

        if (exp > MAX_EXP) {
            // Overflow
            field.setIEEEFlagsBits(DfpField.FLAG_OVERFLOW);
            return DfpField.FLAG_OVERFLOW;
        }

        if (n != 0) {
            // Inexact
            field.setIEEEFlagsBits(DfpField.FLAG_INEXACT);
            return DfpField.FLAG_INEXACT;
        }

        return 0;

    }

    
    public Dfp multiply(final Dfp x) {

        // make sure we don't mix number with different precision
        if (field.getRadixDigits() != x.field.getRadixDigits()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            final Dfp result = newInstance(getZero());
            result.nans = QNAN;
            return dotrap(DfpField.FLAG_INVALID, MULTIPLY_TRAP, x, result);
        }

        Dfp result = newInstance(getZero());

        /* handle special cases */
        if (nans != FINITE || x.nans != FINITE) {
            if (isNaN()) {
                return this;
            }

            if (x.isNaN()) {
                return x;
            }

            if (nans == INFINITE && x.nans == FINITE && x.mant[mant.length-1] != 0) {
                result = newInstance(this);
                result.sign = (byte) (sign * x.sign);
                return result;
            }

            if (x.nans == INFINITE && nans == FINITE && mant[mant.length-1] != 0) {
                result = newInstance(x);
                result.sign = (byte) (sign * x.sign);
                return result;
            }

            if (x.nans == INFINITE && nans == INFINITE) {
                result = newInstance(this);
                result.sign = (byte) (sign * x.sign);
                return result;
            }

            if ( (x.nans == INFINITE && nans == FINITE && mant[mant.length-1] == 0) ||
                    (nans == INFINITE && x.nans == FINITE && x.mant[mant.length-1] == 0) ) {
                field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
                result = newInstance(getZero());
                result.nans = QNAN;
                result = dotrap(DfpField.FLAG_INVALID, MULTIPLY_TRAP, x, result);
                return result;
            }
        }

        int[] product = new int[mant.length*2];  // Big enough to hold even the largest result

        for (int i = 0; i < mant.length; i++) {
            int rh = 0;  // acts as a carry
            for (int j=0; j<mant.length; j++) {
                int r = mant[i] * x.mant[j];    // multiply the 2 digits
                r += product[i+j] + rh;  // add to the product digit with carry in

                rh = r / RADIX;
                product[i+j] = r - rh * RADIX;
            }
            product[i+mant.length] = rh;
        }

        // Find the most sig digit
        int md = mant.length * 2 - 1;  // default, in case result is zero
        for (int i = mant.length * 2 - 1; i >= 0; i--) {
            if (product[i] != 0) {
                md = i;
                break;
            }
        }

        // Copy the digits into the result
        for (int i = 0; i < mant.length; i++) {
            result.mant[mant.length - i - 1] = product[md - i];
        }

        // Fixup the exponent.
        result.exp = exp + x.exp + md - 2 * mant.length + 1;
        result.sign = (byte)((sign == x.sign)?1:-1);

        if (result.mant[mant.length-1] == 0) {
            // if result is zero, set exp to zero
            result.exp = 0;
        }

        final int excp;
        if (md > (mant.length-1)) {
            excp = result.round(product[md-mant.length]);
        } else {
            excp = result.round(0); // has no effect except to check status
        }

        if (excp != 0) {
            result = dotrap(excp, MULTIPLY_TRAP, x, result);
        }

        return result;

    }

    
    public Dfp multiply(final int x) {
        if (x >= 0 && x < RADIX) {
            return multiplyFast(x);
        } else {
            return multiply(newInstance(x));
        }
    }

    
    private Dfp multiplyFast(final int x) {
        Dfp result = newInstance(this);

        /* handle special cases */
        if (nans != FINITE) {
            if (isNaN()) {
                return this;
            }

            if (nans == INFINITE && x != 0) {
                result = newInstance(this);
                return result;
            }

            if (nans == INFINITE && x == 0) {
                field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
                result = newInstance(getZero());
                result.nans = QNAN;
                result = dotrap(DfpField.FLAG_INVALID, MULTIPLY_TRAP, newInstance(getZero()), result);
                return result;
            }
        }

        /* range check x */
        if (x < 0 || x >= RADIX) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            result = newInstance(getZero());
            result.nans = QNAN;
            result = dotrap(DfpField.FLAG_INVALID, MULTIPLY_TRAP, result, result);
            return result;
        }

        int rh = 0;
        for (int i = 0; i < mant.length; i++) {
            final int r = mant[i] * x + rh;
            rh = r / RADIX;
            result.mant[i] = r - rh * RADIX;
        }

        int lostdigit = 0;
        if (rh != 0) {
            lostdigit = result.mant[0];
            result.shiftRight();
            result.mant[mant.length-1] = rh;
        }

        if (result.mant[mant.length-1] == 0) { // if result is zero, set exp to zero
            result.exp = 0;
        }

        final int excp = result.round(lostdigit);
        if (excp != 0) {
            result = dotrap(excp, MULTIPLY_TRAP, result, result);
        }

        return result;
    }

    
    public Dfp divide(Dfp divisor) {
        int dividend[]; // current status of the dividend
        int quotient[]; // quotient
        int remainder[];// remainder
        int qd;         // current quotient digit we're working with
        int nsqd;       // number of significant quotient digits we have
        int trial=0;    // trial quotient digit
        int minadj;     // minimum adjustment
        boolean trialgood; // Flag to indicate a good trail digit
        int md=0;       // most sig digit in result
        int excp;       // exceptions

        // make sure we don't mix number with different precision
        if (field.getRadixDigits() != divisor.field.getRadixDigits()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            final Dfp result = newInstance(getZero());
            result.nans = QNAN;
            return dotrap(DfpField.FLAG_INVALID, DIVIDE_TRAP, divisor, result);
        }

        Dfp result = newInstance(getZero());

        /* handle special cases */
        if (nans != FINITE || divisor.nans != FINITE) {
            if (isNaN()) {
                return this;
            }

            if (divisor.isNaN()) {
                return divisor;
            }

            if (nans == INFINITE && divisor.nans == FINITE) {
                result = newInstance(this);
                result.sign = (byte) (sign * divisor.sign);
                return result;
            }

            if (divisor.nans == INFINITE && nans == FINITE) {
                result = newInstance(getZero());
                result.sign = (byte) (sign * divisor.sign);
                return result;
            }

            if (divisor.nans == INFINITE && nans == INFINITE) {
                field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
                result = newInstance(getZero());
                result.nans = QNAN;
                result = dotrap(DfpField.FLAG_INVALID, DIVIDE_TRAP, divisor, result);
                return result;
            }
        }

        /* Test for divide by zero */
        if (divisor.mant[mant.length-1] == 0) {
            field.setIEEEFlagsBits(DfpField.FLAG_DIV_ZERO);
            result = newInstance(getZero());
            result.sign = (byte) (sign * divisor.sign);
            result.nans = INFINITE;
            result = dotrap(DfpField.FLAG_DIV_ZERO, DIVIDE_TRAP, divisor, result);
            return result;
        }

        dividend = new int[mant.length+1];  // one extra digit needed
        quotient = new int[mant.length+2];  // two extra digits needed 1 for overflow, 1 for rounding
        remainder = new int[mant.length+1]; // one extra digit needed

        /* Initialize our most significant digits to zero */

        dividend[mant.length] = 0;
        quotient[mant.length] = 0;
        quotient[mant.length+1] = 0;
        remainder[mant.length] = 0;

        /* copy our mantissa into the dividend, initialize the
       quotient while we are at it */

        for (int i = 0; i < mant.length; i++) {
            dividend[i] = mant[i];
            quotient[i] = 0;
            remainder[i] = 0;
        }

        /* outer loop.  Once per quotient digit */
        nsqd = 0;
        for (qd = mant.length+1; qd >= 0; qd--) {
            /* Determine outer limits of our quotient digit */

            // r =  most sig 2 digits of dividend
            final int divMsb = dividend[mant.length]*RADIX+dividend[mant.length-1];
            int min = divMsb       / (divisor.mant[mant.length-1]+1);
            int max = (divMsb + 1) / divisor.mant[mant.length-1];

            trialgood = false;
            while (!trialgood) {
                // try the mean
                trial = (min+max)/2;

                /* Multiply by divisor and store as remainder */
                int rh = 0;
                for (int i = 0; i < mant.length + 1; i++) {
                    int dm = (i<mant.length)?divisor.mant[i]:0;
                    final int r = (dm * trial) + rh;
                    rh = r / RADIX;
                    remainder[i] = r - rh * RADIX;
                }

                /* subtract the remainder from the dividend */
                rh = 1;  // carry in to aid the subtraction
                for (int i = 0; i < mant.length + 1; i++) {
                    final int r = ((RADIX-1) - remainder[i]) + dividend[i] + rh;
                    rh = r / RADIX;
                    remainder[i] = r - rh * RADIX;
                }

                /* Lets analyze what we have here */
                if (rh == 0) {
                    // trial is too big -- negative remainder
                    max = trial-1;
                    continue;
                }

                /* find out how far off the remainder is telling us we are */
                minadj = (remainder[mant.length] * RADIX)+remainder[mant.length-1];
                minadj /= divisor.mant[mant.length-1] + 1;

                if (minadj >= 2) {
                    min = trial+minadj;  // update the minimum
                    continue;
                }

                /* May have a good one here, check more thoroughly.  Basically
           its a good one if it is less than the divisor */
                trialgood = false;  // assume false
                for (int i = mant.length - 1; i >= 0; i--) {
                    if (divisor.mant[i] > remainder[i]) {
                        trialgood = true;
                    }
                    if (divisor.mant[i] < remainder[i]) {
                        break;
                    }
                }

                if (remainder[mant.length] != 0) {
                    trialgood = false;
                }

                if (trialgood == false) {
                    min = trial+1;
                }
            }

            /* Great we have a digit! */
            quotient[qd] = trial;
            if (trial != 0 || nsqd != 0) {
                nsqd++;
            }

            if (field.getRoundingMode() == DfpField.RoundingMode.ROUND_DOWN && nsqd == mant.length) {
                // We have enough for this mode
                break;
            }

            if (nsqd > mant.length) {
                // We have enough digits
                break;
            }

            /* move the remainder into the dividend while left shifting */
            dividend[0] = 0;
            for (int i = 0; i < mant.length; i++) {
                dividend[i + 1] = remainder[i];
            }
        }

        /* Find the most sig digit */
        md = mant.length;  // default
        for (int i = mant.length + 1; i >= 0; i--) {
            if (quotient[i] != 0) {
                md = i;
                break;
            }
        }

        /* Copy the digits into the result */
        for (int i=0; i<mant.length; i++) {
            result.mant[mant.length-i-1] = quotient[md-i];
        }

        /* Fixup the exponent. */
        result.exp = exp - divisor.exp + md - mant.length;
        result.sign = (byte) ((sign == divisor.sign) ? 1 : -1);

        if (result.mant[mant.length-1] == 0) { // if result is zero, set exp to zero
            result.exp = 0;
        }

        if (md > (mant.length-1)) {
            excp = result.round(quotient[md-mant.length]);
        } else {
            excp = result.round(0);
        }

        if (excp != 0) {
            result = dotrap(excp, DIVIDE_TRAP, divisor, result);
        }

        return result;
    }

    
    public Dfp divide(int divisor) {

        // Handle special cases
        if (nans != FINITE) {
            if (isNaN()) {
                return this;
            }

            if (nans == INFINITE) {
                return newInstance(this);
            }
        }

        // Test for divide by zero
        if (divisor == 0) {
            field.setIEEEFlagsBits(DfpField.FLAG_DIV_ZERO);
            Dfp result = newInstance(getZero());
            result.sign = sign;
            result.nans = INFINITE;
            result = dotrap(DfpField.FLAG_DIV_ZERO, DIVIDE_TRAP, getZero(), result);
            return result;
        }

        // range check divisor
        if (divisor < 0 || divisor >= RADIX) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            Dfp result = newInstance(getZero());
            result.nans = QNAN;
            result = dotrap(DfpField.FLAG_INVALID, DIVIDE_TRAP, result, result);
            return result;
        }

        Dfp result = newInstance(this);

        int rl = 0;
        for (int i = mant.length-1; i >= 0; i--) {
            final int r = rl*RADIX + result.mant[i];
            final int rh = r / divisor;
            rl = r - rh * divisor;
            result.mant[i] = rh;
        }

        if (result.mant[mant.length-1] == 0) {
            // normalize
            result.shiftLeft();
            final int r = rl * RADIX;        // compute the next digit and put it in
            final int rh = r / divisor;
            rl = r - rh * divisor;
            result.mant[0] = rh;
        }

        final int excp = result.round(rl * RADIX / divisor);  // do the rounding
        if (excp != 0) {
            result = dotrap(excp, DIVIDE_TRAP, result, result);
        }

        return result;

    }

    
    public Dfp reciprocal() {
        return field.getOne().divide(this);
    }

    
    public Dfp sqrt() {

        // check for unusual cases
        if (nans == FINITE && mant[mant.length-1] == 0) {
            // if zero
            return newInstance(this);
        }

        if (nans != FINITE) {
            if (nans == INFINITE && sign == 1) {
                // if positive infinity
                return newInstance(this);
            }

            if (nans == QNAN) {
                return newInstance(this);
            }

            if (nans == SNAN) {
                Dfp result;

                field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
                result = newInstance(this);
                result = dotrap(DfpField.FLAG_INVALID, SQRT_TRAP, null, result);
                return result;
            }
        }

        if (sign == -1) {
            // if negative
            Dfp result;

            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            result = newInstance(this);
            result.nans = QNAN;
            result = dotrap(DfpField.FLAG_INVALID, SQRT_TRAP, null, result);
            return result;
        }

        Dfp x = newInstance(this);

        /* Lets make a reasonable guess as to the size of the square root */
        if (x.exp < -1 || x.exp > 1) {
            x.exp = this.exp / 2;
        }

        /* Coarsely estimate the mantissa */
        switch (x.mant[mant.length-1] / 2000) {
            case 0:
                x.mant[mant.length-1] = x.mant[mant.length-1]/2+1;
                break;
            case 2:
                x.mant[mant.length-1] = 1500;
                break;
            case 3:
                x.mant[mant.length-1] = 2200;
                break;
            default:
                x.mant[mant.length-1] = 3000;
        }

        Dfp dx = newInstance(x);

        /* Now that we have the first pass estimate, compute the rest
       by the formula dx = (y - x*x) / (2x); */

        Dfp px  = getZero();
        Dfp ppx = getZero();
        while (x.unequal(px)) {
            dx = newInstance(x);
            dx.sign = -1;
            dx = dx.add(this.divide(x));
            dx = dx.divide(2);
            ppx = px;
            px = x;
            x = x.add(dx);

            if (x.equals(ppx)) {
                // alternating between two values
                break;
            }

            // if dx is zero, break.  Note testing the most sig digit
            // is a sufficient test since dx is normalized
            if (dx.mant[mant.length-1] == 0) {
                break;
            }
        }

        return x;

    }

    
    @Override
    public String toString() {
        if (nans != FINITE) {
            // if non-finite exceptional cases
            if (nans == INFINITE) {
                return (sign < 0) ? NEG_INFINITY_STRING : POS_INFINITY_STRING;
            } else {
                return NAN_STRING;
            }
        }

        if (exp > mant.length || exp < -1) {
            return dfp2sci();
        }

        return dfp2string();

    }

    
    protected String dfp2sci() {
        char rawdigits[]    = new char[mant.length * 4];
        char outputbuffer[] = new char[mant.length * 4 + 20];
        int p;
        int q;
        int e;
        int ae;
        int shf;

        // Get all the digits
        p = 0;
        for (int i = mant.length - 1; i >= 0; i--) {
            rawdigits[p++] = (char) ((mant[i] / 1000) + '0');
            rawdigits[p++] = (char) (((mant[i] / 100) %10) + '0');
            rawdigits[p++] = (char) (((mant[i] / 10) % 10) + '0');
            rawdigits[p++] = (char) (((mant[i]) % 10) + '0');
        }

        // Find the first non-zero one
        for (p = 0; p < rawdigits.length; p++) {
            if (rawdigits[p] != '0') {
                break;
            }
        }
        shf = p;

        // Now do the conversion
        q = 0;
        if (sign == -1) {
            outputbuffer[q++] = '-';
        }

        if (p != rawdigits.length) {
            // there are non zero digits...
            outputbuffer[q++] = rawdigits[p++];
            outputbuffer[q++] = '.';

            while (p<rawdigits.length) {
                outputbuffer[q++] = rawdigits[p++];
            }
        } else {
            outputbuffer[q++] = '0';
            outputbuffer[q++] = '.';
            outputbuffer[q++] = '0';
            outputbuffer[q++] = 'e';
            outputbuffer[q++] = '0';
            return new String(outputbuffer, 0, 5);
        }

        outputbuffer[q++] = 'e';

        // Find the msd of the exponent

        e = exp * 4 - shf - 1;
        ae = e;
        if (e < 0) {
            ae = -e;
        }

        // Find the largest p such that p < e
        for (p = 1000000000; p > ae; p /= 10) {
            // nothing to do
        }

        if (e < 0) {
            outputbuffer[q++] = '-';
        }

        while (p > 0) {
            outputbuffer[q++] = (char)(ae / p + '0');
            ae %= p;
            p /= 10;
        }

        return new String(outputbuffer, 0, q);

    }

    
    protected String dfp2string() {
        char buffer[] = new char[mant.length*4 + 20];
        int p = 1;
        int q;
        int e = exp;
        boolean pointInserted = false;

        buffer[0] = ' ';

        if (e <= 0) {
            buffer[p++] = '0';
            buffer[p++] = '.';
            pointInserted = true;
        }

        while (e < 0) {
            buffer[p++] = '0';
            buffer[p++] = '0';
            buffer[p++] = '0';
            buffer[p++] = '0';
            e++;
        }

        for (int i = mant.length - 1; i >= 0; i--) {
            buffer[p++] = (char) ((mant[i] / 1000) + '0');
            buffer[p++] = (char) (((mant[i] / 100) % 10) + '0');
            buffer[p++] = (char) (((mant[i] / 10) % 10) + '0');
            buffer[p++] = (char) (((mant[i]) % 10) + '0');
            if (--e == 0) {
                buffer[p++] = '.';
                pointInserted = true;
            }
        }

        while (e > 0) {
            buffer[p++] = '0';
            buffer[p++] = '0';
            buffer[p++] = '0';
            buffer[p++] = '0';
            e--;
        }

        if (!pointInserted) {
            // Ensure we have a radix point!
            buffer[p++] = '.';
        }

        // Suppress leading zeros
        q = 1;
        while (buffer[q] == '0') {
            q++;
        }
        if (buffer[q] == '.') {
            q--;
        }

        // Suppress trailing zeros
        while (buffer[p-1] == '0') {
            p--;
        }

        // Insert sign
        if (sign < 0) {
            buffer[--q] = '-';
        }

        return new String(buffer, q, p - q);

    }

    
    public Dfp dotrap(int type, String what, Dfp oper, Dfp result) {
        Dfp def = result;

        switch (type) {
            case DfpField.FLAG_INVALID:
                def = newInstance(getZero());
                def.sign = result.sign;
                def.nans = QNAN;
                break;

            case DfpField.FLAG_DIV_ZERO:
                if (nans == FINITE && mant[mant.length-1] != 0) {
                    // normal case, we are finite, non-zero
                    def = newInstance(getZero());
                    def.sign = (byte)(sign*oper.sign);
                    def.nans = INFINITE;
                }

                if (nans == FINITE && mant[mant.length-1] == 0) {
                    //  0/0
                    def = newInstance(getZero());
                    def.nans = QNAN;
                }

                if (nans == INFINITE || nans == QNAN) {
                    def = newInstance(getZero());
                    def.nans = QNAN;
                }

                if (nans == INFINITE || nans == SNAN) {
                    def = newInstance(getZero());
                    def.nans = QNAN;
                }
                break;

            case DfpField.FLAG_UNDERFLOW:
                if ( (result.exp+mant.length) < MIN_EXP) {
                    def = newInstance(getZero());
                    def.sign = result.sign;
                } else {
                    def = newInstance(result);  // gradual underflow
                }
                result.exp += ERR_SCALE;
                break;

            case DfpField.FLAG_OVERFLOW:
                result.exp -= ERR_SCALE;
                def = newInstance(getZero());
                def.sign = result.sign;
                def.nans = INFINITE;
                break;

            default: def = result; break;
        }

        return trap(type, what, oper, def, result);

    }

    
    protected Dfp trap(int type, String what, Dfp oper, Dfp def, Dfp result) {
        return def;
    }

    
    public int classify() {
        return nans;
    }

    
    public static Dfp copysign(final Dfp x, final Dfp y) {
        Dfp result = x.newInstance(x);
        result.sign = y.sign;
        return result;
    }

    
    public Dfp nextAfter(final Dfp x) {

        // make sure we don't mix number with different precision
        if (field.getRadixDigits() != x.field.getRadixDigits()) {
            field.setIEEEFlagsBits(DfpField.FLAG_INVALID);
            final Dfp result = newInstance(getZero());
            result.nans = QNAN;
            return dotrap(DfpField.FLAG_INVALID, NEXT_AFTER_TRAP, x, result);
        }

        // if this is greater than x
        boolean up = false;
        if (this.lessThan(x)) {
            up = true;
        }

        if (compare(this, x) == 0) {
            return newInstance(x);
        }

        if (lessThan(getZero())) {
            up = !up;
        }

        final Dfp inc;
        Dfp result;
        if (up) {
            inc = newInstance(getOne());
            inc.exp = this.exp-mant.length+1;
            inc.sign = this.sign;

            if (this.equals(getZero())) {
                inc.exp = MIN_EXP-mant.length;
            }

            result = add(inc);
        } else {
            inc = newInstance(getOne());
            inc.exp = this.exp;
            inc.sign = this.sign;

            if (this.equals(inc)) {
                inc.exp = this.exp-mant.length;
            } else {
                inc.exp = this.exp-mant.length+1;
            }

            if (this.equals(getZero())) {
                inc.exp = MIN_EXP-mant.length;
            }

            result = this.subtract(inc);
        }

        if (result.classify() == INFINITE && this.classify() != INFINITE) {
            field.setIEEEFlagsBits(DfpField.FLAG_INEXACT);
            result = dotrap(DfpField.FLAG_INEXACT, NEXT_AFTER_TRAP, x, result);
        }

        if (result.equals(getZero()) && this.equals(getZero()) == false) {
            field.setIEEEFlagsBits(DfpField.FLAG_INEXACT);
            result = dotrap(DfpField.FLAG_INEXACT, NEXT_AFTER_TRAP, x, result);
        }

        return result;

    }

    
    public double toDouble() {

        if (isInfinite()) {
            if (lessThan(getZero())) {
                return Double.NEGATIVE_INFINITY;
            } else {
                return Double.POSITIVE_INFINITY;
            }
        }

        if (isNaN()) {
            return Double.NaN;
        }

        Dfp y = this;
        boolean negate = false;
        int cmp0 = compare(this, getZero());
        if (cmp0 == 0) {
            return sign < 0 ? -0.0 : +0.0;
        } else if (cmp0 < 0) {
            y = negate();
            negate = true;
        }

        /* Find the exponent, first estimate by integer log10, then adjust.
         Should be faster than doing a natural logarithm.  */
        int exponent = (int)(y.intLog10() * 3.32);
        if (exponent < 0) {
            exponent--;
        }

        Dfp tempDfp = DfpMath.pow(getTwo(), exponent);
        while (tempDfp.lessThan(y) || tempDfp.equals(y)) {
            tempDfp = tempDfp.multiply(2);
            exponent++;
        }
        exponent--;

        /* We have the exponent, now work on the mantissa */

        y = y.divide(DfpMath.pow(getTwo(), exponent));
        if (exponent > -1023) {
            y = y.subtract(getOne());
        }

        if (exponent < -1074) {
            return 0;
        }

        if (exponent > 1023) {
            return negate ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        }


        y = y.multiply(newInstance(4503599627370496l)).rint();
        String str = y.toString();
        str = str.substring(0, str.length()-1);
        long mantissa = Long.parseLong(str);

        if (mantissa == 4503599627370496L) {
            // Handle special case where we round up to next power of two
            mantissa = 0;
            exponent++;
        }

        /* Its going to be subnormal, so make adjustments */
        if (exponent <= -1023) {
            exponent--;
        }

        while (exponent < -1023) {
            exponent++;
            mantissa >>>= 1;
        }

        long bits = mantissa | ((exponent + 1023L) << 52);
        double x = Double.longBitsToDouble(bits);

        if (negate) {
            x = -x;
        }

        return x;

    }

    
    public double[] toSplitDouble() {
        double split[] = new double[2];
        long mask = 0xffffffffc0000000L;

        split[0] = Double.longBitsToDouble(Double.doubleToLongBits(toDouble()) & mask);
        split[1] = subtract(newInstance(split[0])).toDouble();

        return split;
    }

    
    public double getReal() {
        return toDouble();
    }

    
    public Dfp add(final double a) {
        return add(newInstance(a));
    }

    
    public Dfp subtract(final double a) {
        return subtract(newInstance(a));
    }

    
    public Dfp multiply(final double a) {
        return multiply(newInstance(a));
    }

    
    public Dfp divide(final double a) {
        return divide(newInstance(a));
    }

    
    public Dfp remainder(final double a) {
        return remainder(newInstance(a));
    }

    
    public long round() {
        return FastMath.round(toDouble());
    }

    
    public Dfp signum() {
        if (isNaN() || isZero()) {
            return this;
        } else {
            return newInstance(sign > 0 ? +1 : -1);
        }
    }

    
    public Dfp copySign(final Dfp s) {
        if ((sign >= 0 && s.sign >= 0) || (sign < 0 && s.sign < 0)) { // Sign is currently OK
            return this;
        }
        return negate(); // flip sign
    }

    
    public Dfp copySign(final double s) {
        long sb = Double.doubleToLongBits(s);
        if ((sign >= 0 && sb >= 0) || (sign < 0 && sb < 0)) { // Sign is currently OK
            return this;
        }
        return negate(); // flip sign
    }

    
    public Dfp scalb(final int n) {
        return multiply(DfpMath.pow(getTwo(), n));
    }

    
    public Dfp hypot(final Dfp y) {
        return multiply(this).add(y.multiply(y)).sqrt();
    }

    
    public Dfp cbrt() {
        return rootN(3);
    }

    
    public Dfp rootN(final int n) {
        return (sign >= 0) ?
               DfpMath.pow(this, getOne().divide(n)) :
               DfpMath.pow(negate(), getOne().divide(n)).negate();
    }

    
    public Dfp pow(final double p) {
        return DfpMath.pow(this, newInstance(p));
    }

    
    public Dfp pow(final int n) {
        return DfpMath.pow(this, n);
    }

    
    public Dfp pow(final Dfp e) {
        return DfpMath.pow(this, e);
    }

    
    public Dfp exp() {
        return DfpMath.exp(this);
    }

    
    public Dfp expm1() {
        return DfpMath.exp(this).subtract(getOne());
    }

    
    public Dfp log() {
        return DfpMath.log(this);
    }

    
    public Dfp log1p() {
        return DfpMath.log(this.add(getOne()));
    }

//  TODO: deactivate this implementation (and return type) in 4.0
    
    @Deprecated
    public int log10()  {
        return intLog10();
    }

//    TODO: activate this implementation (and return type) in 4.0
//    
//    public Dfp log10() {
//        return DfpMath.log(this).divide(DfpMath.log(newInstance(10)));
//    }

    
    public Dfp cos() {
        return DfpMath.cos(this);
    }

    
    public Dfp sin() {
        return DfpMath.sin(this);
    }

    
    public Dfp tan() {
        return DfpMath.tan(this);
    }

    
    public Dfp acos() {
        return DfpMath.acos(this);
    }

    
    public Dfp asin() {
        return DfpMath.asin(this);
    }

    
    public Dfp atan() {
        return DfpMath.atan(this);
    }

    
    public Dfp atan2(final Dfp x)
        throws DimensionMismatchException {

        // compute r = sqrt(x^2+y^2)
        final Dfp r = x.multiply(x).add(multiply(this)).sqrt();

        if (x.sign >= 0) {

            // compute atan2(y, x) = 2 atan(y / (r + x))
            return getTwo().multiply(divide(r.add(x)).atan());

        } else {

            // compute atan2(y, x) = +/- pi - 2 atan(y / (r - x))
            final Dfp tmp = getTwo().multiply(divide(r.subtract(x)).atan());
            final Dfp pmPi = newInstance((tmp.sign <= 0) ? -FastMath.PI : FastMath.PI);
            return pmPi.subtract(tmp);

        }

    }

    
    public Dfp cosh() {
        return DfpMath.exp(this).add(DfpMath.exp(negate())).divide(2);
    }

    
    public Dfp sinh() {
        return DfpMath.exp(this).subtract(DfpMath.exp(negate())).divide(2);
    }

    
    public Dfp tanh() {
        final Dfp ePlus  = DfpMath.exp(this);
        final Dfp eMinus = DfpMath.exp(negate());
        return ePlus.subtract(eMinus).divide(ePlus.add(eMinus));
    }

    
    public Dfp acosh() {
        return multiply(this).subtract(getOne()).sqrt().add(this).log();
    }

    
    public Dfp asinh() {
        return multiply(this).add(getOne()).sqrt().add(this).log();
    }

    
    public Dfp atanh() {
        return getOne().add(this).divide(getOne().subtract(this)).log().divide(2);
    }

    
    public Dfp linearCombination(final Dfp[] a, final Dfp[] b)
        throws DimensionMismatchException {
        if (a.length != b.length) {
            throw new DimensionMismatchException(a.length, b.length);
        }
        Dfp r = getZero();
        for (int i = 0; i < a.length; ++i) {
            r = r.add(a[i].multiply(b[i]));
        }
        return r;
    }

    
    public Dfp linearCombination(final double[] a, final Dfp[] b)
        throws DimensionMismatchException {
        if (a.length != b.length) {
            throw new DimensionMismatchException(a.length, b.length);
        }
        Dfp r = getZero();
        for (int i = 0; i < a.length; ++i) {
            r = r.add(b[i].multiply(a[i]));
        }
        return r;
    }

    
    public Dfp linearCombination(final Dfp a1, final Dfp b1, final Dfp a2, final Dfp b2) {
        return a1.multiply(b1).add(a2.multiply(b2));
    }

    
    public Dfp linearCombination(final double a1, final Dfp b1, final double a2, final Dfp b2) {
        return b1.multiply(a1).add(b2.multiply(a2));
    }

    
    public Dfp linearCombination(final Dfp a1, final Dfp b1,
                                 final Dfp a2, final Dfp b2,
                                 final Dfp a3, final Dfp b3) {
        return a1.multiply(b1).add(a2.multiply(b2)).add(a3.multiply(b3));
    }

    
    public Dfp linearCombination(final double a1, final Dfp b1,
                                 final double a2, final Dfp b2,
                                 final double a3, final Dfp b3) {
        return b1.multiply(a1).add(b2.multiply(a2)).add(b3.multiply(a3));
    }

    
    public Dfp linearCombination(final Dfp a1, final Dfp b1, final Dfp a2, final Dfp b2,
                                 final Dfp a3, final Dfp b3, final Dfp a4, final Dfp b4) {
        return a1.multiply(b1).add(a2.multiply(b2)).add(a3.multiply(b3)).add(a4.multiply(b4));
    }

    
    public Dfp linearCombination(final double a1, final Dfp b1, final double a2, final Dfp b2,
                                 final double a3, final Dfp b3, final double a4, final Dfp b4) {
        return b1.multiply(a1).add(b2.multiply(a2)).add(b3.multiply(a3)).add(b4.multiply(a4));
    }

}
