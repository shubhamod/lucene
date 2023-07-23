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

package org.apache.lucene.util.hnsw.math.util;

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NotFiniteNumberException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.Localizable;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public final class MathUtils {
    
    public static final double TWO_PI = 2 * FastMath.PI;

    
    public static final double PI_SQUARED = FastMath.PI * FastMath.PI;


    
    private MathUtils() {}


    
    public static int hash(double value) {
        return new Double(value).hashCode();
    }

    
    public static boolean equals(double x, double y) {
        return new Double(x).equals(new Double(y));
    }

    
    public static int hash(double[] value) {
        return Arrays.hashCode(value);
    }

    
     public static double normalizeAngle(double a, double center) {
         return a - TWO_PI * FastMath.floor((a + FastMath.PI - center) / TWO_PI);
     }

     
     public static <T extends RealFieldElement<T>> T max(final T e1, final T e2) {
         return e1.subtract(e2).getReal() >= 0 ? e1 : e2;
     }

     
     public static <T extends RealFieldElement<T>> T min(final T e1, final T e2) {
         return e1.subtract(e2).getReal() >= 0 ? e2 : e1;
     }

    
    public static double reduce(double a,
                                double period,
                                double offset) {
        final double p = FastMath.abs(period);
        return a - p * FastMath.floor((a - offset) / p) - offset;
    }

    
    public static byte copySign(byte magnitude, byte sign)
        throws MathArithmeticException {
        if ((magnitude >= 0 && sign >= 0) ||
            (magnitude < 0 && sign < 0)) { // Sign is OK.
            return magnitude;
        } else if (sign >= 0 &&
                   magnitude == Byte.MIN_VALUE) {
            throw new MathArithmeticException(LocalizedFormats.OVERFLOW);
        } else {
            return (byte) -magnitude; // Flip sign.
        }
    }

    
    public static short copySign(short magnitude, short sign)
            throws MathArithmeticException {
        if ((magnitude >= 0 && sign >= 0) ||
            (magnitude < 0 && sign < 0)) { // Sign is OK.
            return magnitude;
        } else if (sign >= 0 &&
                   magnitude == Short.MIN_VALUE) {
            throw new MathArithmeticException(LocalizedFormats.OVERFLOW);
        } else {
            return (short) -magnitude; // Flip sign.
        }
    }

    
    public static int copySign(int magnitude, int sign)
            throws MathArithmeticException {
        if ((magnitude >= 0 && sign >= 0) ||
            (magnitude < 0 && sign < 0)) { // Sign is OK.
            return magnitude;
        } else if (sign >= 0 &&
                   magnitude == Integer.MIN_VALUE) {
            throw new MathArithmeticException(LocalizedFormats.OVERFLOW);
        } else {
            return -magnitude; // Flip sign.
        }
    }

    
    public static long copySign(long magnitude, long sign)
        throws MathArithmeticException {
        if ((magnitude >= 0 && sign >= 0) ||
            (magnitude < 0 && sign < 0)) { // Sign is OK.
            return magnitude;
        } else if (sign >= 0 &&
                   magnitude == Long.MIN_VALUE) {
            throw new MathArithmeticException(LocalizedFormats.OVERFLOW);
        } else {
            return -magnitude; // Flip sign.
        }
    }
    
    public static void checkFinite(final double x)
        throws NotFiniteNumberException {
        if (Double.isInfinite(x) || Double.isNaN(x)) {
            throw new NotFiniteNumberException(x);
        }
    }

    
    public static void checkFinite(final double[] val)
        throws NotFiniteNumberException {
        for (int i = 0; i < val.length; i++) {
            final double x = val[i];
            if (Double.isInfinite(x) || Double.isNaN(x)) {
                throw new NotFiniteNumberException(LocalizedFormats.ARRAY_ELEMENT, x, i);
            }
        }
    }

    
    public static void checkNotNull(Object o,
                                    Localizable pattern,
                                    Object ... args)
        throws NullArgumentException {
        if (o == null) {
            throw new NullArgumentException(pattern, args);
        }
    }

    
    public static void checkNotNull(Object o)
        throws NullArgumentException {
        if (o == null) {
            throw new NullArgumentException();
        }
    }
}
