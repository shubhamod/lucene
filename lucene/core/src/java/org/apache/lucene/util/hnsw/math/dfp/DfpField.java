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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;


public class DfpField implements Field<Dfp> {

    
    public enum RoundingMode {

        
        ROUND_DOWN,

        
        ROUND_UP,

        
        ROUND_HALF_UP,

        
        ROUND_HALF_DOWN,

        
        ROUND_HALF_EVEN,

        
        ROUND_HALF_ODD,

        
        ROUND_CEIL,

        
        ROUND_FLOOR;

    }

    
    public static final int FLAG_INVALID   =  1;

    
    public static final int FLAG_DIV_ZERO  =  2;

    
    public static final int FLAG_OVERFLOW  =  4;

    
    public static final int FLAG_UNDERFLOW =  8;

    
    public static final int FLAG_INEXACT   = 16;

    
    private static String sqr2String;

    // Note: the static strings are set up (once) by the ctor and @GuardedBy("DfpField.class")

    
    private static String sqr2ReciprocalString;

    
    private static String sqr3String;

    
    private static String sqr3ReciprocalString;

    
    private static String piString;

    
    private static String eString;

    
    private static String ln2String;

    
    private static String ln5String;

    
    private static String ln10String;

    
    private final int radixDigits;

    
    private final Dfp zero;

    
    private final Dfp one;

    
    private final Dfp two;

    
    private final Dfp sqr2;

    
    private final Dfp[] sqr2Split;

    
    private final Dfp sqr2Reciprocal;

    
    private final Dfp sqr3;

    
    private final Dfp sqr3Reciprocal;

    
    private final Dfp pi;

    
    private final Dfp[] piSplit;

    
    private final Dfp e;

    
    private final Dfp[] eSplit;

    
    private final Dfp ln2;

    
    private final Dfp[] ln2Split;

    
    private final Dfp ln5;

    
    private final Dfp[] ln5Split;

    
    private final Dfp ln10;

    
    private RoundingMode rMode;

    
    private int ieeeFlags;

    
    public DfpField(final int decimalDigits) {
        this(decimalDigits, true);
    }

    
    private DfpField(final int decimalDigits, final boolean computeConstants) {

        this.radixDigits = (decimalDigits < 13) ? 4 : (decimalDigits + 3) / 4;
        this.rMode       = RoundingMode.ROUND_HALF_EVEN;
        this.ieeeFlags   = 0;
        this.zero        = new Dfp(this, 0);
        this.one         = new Dfp(this, 1);
        this.two         = new Dfp(this, 2);

        if (computeConstants) {
            // set up transcendental constants
            synchronized (DfpField.class) {

                // as a heuristic to circumvent Table-Maker's Dilemma, we set the string
                // representation of the constants to be at least 3 times larger than the
                // number of decimal digits, also as an attempt to really compute these
                // constants only once, we set a minimum number of digits
                computeStringConstants((decimalDigits < 67) ? 200 : (3 * decimalDigits));

                // set up the constants at current field accuracy
                sqr2           = new Dfp(this, sqr2String);
                sqr2Split      = split(sqr2String);
                sqr2Reciprocal = new Dfp(this, sqr2ReciprocalString);
                sqr3           = new Dfp(this, sqr3String);
                sqr3Reciprocal = new Dfp(this, sqr3ReciprocalString);
                pi             = new Dfp(this, piString);
                piSplit        = split(piString);
                e              = new Dfp(this, eString);
                eSplit         = split(eString);
                ln2            = new Dfp(this, ln2String);
                ln2Split       = split(ln2String);
                ln5            = new Dfp(this, ln5String);
                ln5Split       = split(ln5String);
                ln10           = new Dfp(this, ln10String);

            }
        } else {
            // dummy settings for unused constants
            sqr2           = null;
            sqr2Split      = null;
            sqr2Reciprocal = null;
            sqr3           = null;
            sqr3Reciprocal = null;
            pi             = null;
            piSplit        = null;
            e              = null;
            eSplit         = null;
            ln2            = null;
            ln2Split       = null;
            ln5            = null;
            ln5Split       = null;
            ln10           = null;
        }

    }

    
    public int getRadixDigits() {
        return radixDigits;
    }

    
    public void setRoundingMode(final RoundingMode mode) {
        rMode = mode;
    }

    
    public RoundingMode getRoundingMode() {
        return rMode;
    }

    
    public int getIEEEFlags() {
        return ieeeFlags;
    }

    
    public void clearIEEEFlags() {
        ieeeFlags = 0;
    }

    
    public void setIEEEFlags(final int flags) {
        ieeeFlags = flags & (FLAG_INVALID | FLAG_DIV_ZERO | FLAG_OVERFLOW | FLAG_UNDERFLOW | FLAG_INEXACT);
    }

    
    public void setIEEEFlagsBits(final int bits) {
        ieeeFlags |= bits & (FLAG_INVALID | FLAG_DIV_ZERO | FLAG_OVERFLOW | FLAG_UNDERFLOW | FLAG_INEXACT);
    }

    
    public Dfp newDfp() {
        return new Dfp(this);
    }

    
    public Dfp newDfp(final byte x) {
        return new Dfp(this, x);
    }

    
    public Dfp newDfp(final int x) {
        return new Dfp(this, x);
    }

    
    public Dfp newDfp(final long x) {
        return new Dfp(this, x);
    }

    
    public Dfp newDfp(final double x) {
        return new Dfp(this, x);
    }

    
    public Dfp newDfp(Dfp d) {
        return new Dfp(d);
    }

    
    public Dfp newDfp(final String s) {
        return new Dfp(this, s);
    }

    
    public Dfp newDfp(final byte sign, final byte nans) {
        return new Dfp(this, sign, nans);
    }

    
    public Dfp getZero() {
        return zero;
    }

    
    public Dfp getOne() {
        return one;
    }

    
    public Class<? extends FieldElement<Dfp>> getRuntimeClass() {
        return Dfp.class;
    }

    
    public Dfp getTwo() {
        return two;
    }

    
    public Dfp getSqr2() {
        return sqr2;
    }

    
    public Dfp[] getSqr2Split() {
        return sqr2Split.clone();
    }

    
    public Dfp getSqr2Reciprocal() {
        return sqr2Reciprocal;
    }

    
    public Dfp getSqr3() {
        return sqr3;
    }

    
    public Dfp getSqr3Reciprocal() {
        return sqr3Reciprocal;
    }

    
    public Dfp getPi() {
        return pi;
    }

    
    public Dfp[] getPiSplit() {
        return piSplit.clone();
    }

    
    public Dfp getE() {
        return e;
    }

    
    public Dfp[] getESplit() {
        return eSplit.clone();
    }

    
    public Dfp getLn2() {
        return ln2;
    }

    
    public Dfp[] getLn2Split() {
        return ln2Split.clone();
    }

    
    public Dfp getLn5() {
        return ln5;
    }

    
    public Dfp[] getLn5Split() {
        return ln5Split.clone();
    }

    
    public Dfp getLn10() {
        return ln10;
    }

    
    private Dfp[] split(final String a) {
      Dfp result[] = new Dfp[2];
      boolean leading = true;
      int sp = 0;
      int sig = 0;

      char[] buf = new char[a.length()];

      for (int i = 0; i < buf.length; i++) {
        buf[i] = a.charAt(i);

        if (buf[i] >= '1' && buf[i] <= '9') {
            leading = false;
        }

        if (buf[i] == '.') {
          sig += (400 - sig) % 4;
          leading = false;
        }

        if (sig == (radixDigits / 2) * 4) {
          sp = i;
          break;
        }

        if (buf[i] >= '0' && buf[i] <= '9' && !leading) {
            sig ++;
        }
      }

      result[0] = new Dfp(this, new String(buf, 0, sp));

      for (int i = 0; i < buf.length; i++) {
        buf[i] = a.charAt(i);
        if (buf[i] >= '0' && buf[i] <= '9' && i < sp) {
            buf[i] = '0';
        }
      }

      result[1] = new Dfp(this, new String(buf));

      return result;

    }

    
    private static void computeStringConstants(final int highPrecisionDecimalDigits) {
        if (sqr2String == null || sqr2String.length() < highPrecisionDecimalDigits - 3) {

            // recompute the string representation of the transcendental constants
            final DfpField highPrecisionField = new DfpField(highPrecisionDecimalDigits, false);
            final Dfp highPrecisionOne        = new Dfp(highPrecisionField, 1);
            final Dfp highPrecisionTwo        = new Dfp(highPrecisionField, 2);
            final Dfp highPrecisionThree      = new Dfp(highPrecisionField, 3);

            final Dfp highPrecisionSqr2 = highPrecisionTwo.sqrt();
            sqr2String           = highPrecisionSqr2.toString();
            sqr2ReciprocalString = highPrecisionOne.divide(highPrecisionSqr2).toString();

            final Dfp highPrecisionSqr3 = highPrecisionThree.sqrt();
            sqr3String           = highPrecisionSqr3.toString();
            sqr3ReciprocalString = highPrecisionOne.divide(highPrecisionSqr3).toString();

            piString   = computePi(highPrecisionOne, highPrecisionTwo, highPrecisionThree).toString();
            eString    = computeExp(highPrecisionOne, highPrecisionOne).toString();
            ln2String  = computeLn(highPrecisionTwo, highPrecisionOne, highPrecisionTwo).toString();
            ln5String  = computeLn(new Dfp(highPrecisionField, 5),  highPrecisionOne, highPrecisionTwo).toString();
            ln10String = computeLn(new Dfp(highPrecisionField, 10), highPrecisionOne, highPrecisionTwo).toString();

        }
    }

    
    private static Dfp computePi(final Dfp one, final Dfp two, final Dfp three) {

        Dfp sqrt2   = two.sqrt();
        Dfp yk      = sqrt2.subtract(one);
        Dfp four    = two.add(two);
        Dfp two2kp3 = two;
        Dfp ak      = two.multiply(three.subtract(two.multiply(sqrt2)));

        // The formula converges quartically. This means the number of correct
        // digits is multiplied by 4 at each iteration! Five iterations are
        // sufficient for about 160 digits, eight iterations give about
        // 10000 digits (this has been checked) and 20 iterations more than
        // 160 billions of digits (this has NOT been checked).
        // So the limit here is considered sufficient for most purposes ...
        for (int i = 1; i < 20; i++) {
            final Dfp ykM1 = yk;

            final Dfp y2         = yk.multiply(yk);
            final Dfp oneMinusY4 = one.subtract(y2.multiply(y2));
            final Dfp s          = oneMinusY4.sqrt().sqrt();
            yk = one.subtract(s).divide(one.add(s));

            two2kp3 = two2kp3.multiply(four);

            final Dfp p = one.add(yk);
            final Dfp p2 = p.multiply(p);
            ak = ak.multiply(p2.multiply(p2)).subtract(two2kp3.multiply(yk).multiply(one.add(yk).add(yk.multiply(yk))));

            if (yk.equals(ykM1)) {
                break;
            }
        }

        return one.divide(ak);

    }

    
    public static Dfp computeExp(final Dfp a, final Dfp one) {

        Dfp y  = new Dfp(one);
        Dfp py = new Dfp(one);
        Dfp f  = new Dfp(one);
        Dfp fi = new Dfp(one);
        Dfp x  = new Dfp(one);

        for (int i = 0; i < 10000; i++) {
            x = x.multiply(a);
            y = y.add(x.divide(f));
            fi = fi.add(one);
            f = f.multiply(fi);
            if (y.equals(py)) {
                break;
            }
            py = new Dfp(y);
        }

        return y;

    }


    

    public static Dfp computeLn(final Dfp a, final Dfp one, final Dfp two) {

        int den = 1;
        Dfp x = a.add(new Dfp(a.getField(), -1)).divide(a.add(one));

        Dfp y = new Dfp(x);
        Dfp num = new Dfp(x);
        Dfp py = new Dfp(y);
        for (int i = 0; i < 10000; i++) {
            num = num.multiply(x);
            num = num.multiply(x);
            den += 2;
            Dfp t = num.divide(den);
            y = y.add(t);
            if (y.equals(py)) {
                break;
            }
            py = new Dfp(y);
        }

        return y.multiply(two);

    }

}
