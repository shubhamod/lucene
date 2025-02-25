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

import java.math.BigInteger;

import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.util.Localizable;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public final class ArithmeticUtils {

    
    private ArithmeticUtils() {
        super();
    }

    
    public static int addAndCheck(int x, int y)
            throws MathArithmeticException {
        long s = (long)x + (long)y;
        if (s < Integer.MIN_VALUE || s > Integer.MAX_VALUE) {
            throw new MathArithmeticException(LocalizedFormats.OVERFLOW_IN_ADDITION, x, y);
        }
        return (int)s;
    }

    
    public static long addAndCheck(long a, long b) throws MathArithmeticException {
        return addAndCheck(a, b, LocalizedFormats.OVERFLOW_IN_ADDITION);
    }

    
    @Deprecated
    public static long binomialCoefficient(final int n, final int k)
        throws NotPositiveException, NumberIsTooLargeException, MathArithmeticException {
       return CombinatoricsUtils.binomialCoefficient(n, k);
    }

    
    @Deprecated
    public static double binomialCoefficientDouble(final int n, final int k)
        throws NotPositiveException, NumberIsTooLargeException, MathArithmeticException {
        return CombinatoricsUtils.binomialCoefficientDouble(n, k);
    }

    
    @Deprecated
    public static double binomialCoefficientLog(final int n, final int k)
        throws NotPositiveException, NumberIsTooLargeException, MathArithmeticException {
        return CombinatoricsUtils.binomialCoefficientLog(n, k);
    }

    
    @Deprecated
    public static long factorial(final int n) throws NotPositiveException, MathArithmeticException {
        return CombinatoricsUtils.factorial(n);
    }

    
    @Deprecated
    public static double factorialDouble(final int n) throws NotPositiveException {
         return CombinatoricsUtils.factorialDouble(n);
    }

    
    @Deprecated
    public static double factorialLog(final int n) throws NotPositiveException {
        return CombinatoricsUtils.factorialLog(n);
    }

    
    public static int gcd(int p, int q) throws MathArithmeticException {
        int a = p;
        int b = q;
        if (a == 0 ||
            b == 0) {
            if (a == Integer.MIN_VALUE ||
                b == Integer.MIN_VALUE) {
                throw new MathArithmeticException(LocalizedFormats.GCD_OVERFLOW_32_BITS,
                                                  p, q);
            }
            return FastMath.abs(a + b);
        }

        long al = a;
        long bl = b;
        boolean useLong = false;
        if (a < 0) {
            if(Integer.MIN_VALUE == a) {
                useLong = true;
            } else {
                a = -a;
            }
            al = -al;
        }
        if (b < 0) {
            if (Integer.MIN_VALUE == b) {
                useLong = true;
            } else {
                b = -b;
            }
            bl = -bl;
        }
        if (useLong) {
            if(al == bl) {
                throw new MathArithmeticException(LocalizedFormats.GCD_OVERFLOW_32_BITS,
                                                  p, q);
            }
            long blbu = bl;
            bl = al;
            al = blbu % al;
            if (al == 0) {
                if (bl > Integer.MAX_VALUE) {
                    throw new MathArithmeticException(LocalizedFormats.GCD_OVERFLOW_32_BITS,
                                                      p, q);
                }
                return (int) bl;
            }
            blbu = bl;

            // Now "al" and "bl" fit in an "int".
            b = (int) al;
            a = (int) (blbu % al);
        }

        return gcdPositive(a, b);
    }

    
    private static int gcdPositive(int a, int b) {
        if (a == 0) {
            return b;
        }
        else if (b == 0) {
            return a;
        }

        // Make "a" and "b" odd, keeping track of common power of 2.
        final int aTwos = Integer.numberOfTrailingZeros(a);
        a >>= aTwos;
        final int bTwos = Integer.numberOfTrailingZeros(b);
        b >>= bTwos;
        final int shift = FastMath.min(aTwos, bTwos);

        // "a" and "b" are positive.
        // If a > b then "gdc(a, b)" is equal to "gcd(a - b, b)".
        // If a < b then "gcd(a, b)" is equal to "gcd(b - a, a)".
        // Hence, in the successive iterations:
        //  "a" becomes the absolute difference of the current values,
        //  "b" becomes the minimum of the current values.
        while (a != b) {
            final int delta = a - b;
            b = Math.min(a, b);
            a = Math.abs(delta);

            // Remove any power of 2 in "a" ("b" is guaranteed to be odd).
            a >>= Integer.numberOfTrailingZeros(a);
        }

        // Recover the common power of 2.
        return a << shift;
    }

    
    public static long gcd(final long p, final long q) throws MathArithmeticException {
        long u = p;
        long v = q;
        if ((u == 0) || (v == 0)) {
            if ((u == Long.MIN_VALUE) || (v == Long.MIN_VALUE)){
                throw new MathArithmeticException(LocalizedFormats.GCD_OVERFLOW_64_BITS,
                                                  p, q);
            }
            return FastMath.abs(u) + FastMath.abs(v);
        }
        // keep u and v negative, as negative integers range down to
        // -2^63, while positive numbers can only be as large as 2^63-1
        // (i.e. we can't necessarily negate a negative number without
        // overflow)
        /* assert u!=0 && v!=0; */
        if (u > 0) {
            u = -u;
        } // make u negative
        if (v > 0) {
            v = -v;
        } // make v negative
        // B1. [Find power of 2]
        int k = 0;
        while ((u & 1) == 0 && (v & 1) == 0 && k < 63) { // while u and v are
                                                            // both even...
            u /= 2;
            v /= 2;
            k++; // cast out twos.
        }
        if (k == 63) {
            throw new MathArithmeticException(LocalizedFormats.GCD_OVERFLOW_64_BITS,
                                              p, q);
        }
        // B2. Initialize: u and v have been divided by 2^k and at least
        // one is odd.
        long t = ((u & 1) == 1) ? v : -(u / 2)/* B3 */;
        // t negative: u was odd, v may be even (t replaces v)
        // t positive: u was even, v is odd (t replaces u)
        do {
            /* assert u<0 && v<0; */
            // B4/B3: cast out twos from t.
            while ((t & 1) == 0) { // while t is even..
                t /= 2; // cast out twos
            }
            // B5 [reset max(u,v)]
            if (t > 0) {
                u = -t;
            } else {
                v = t;
            }
            // B6/B3. at this point both u and v should be odd.
            t = (v - u) / 2;
            // |u| larger: t positive (replace u)
            // |v| larger: t negative (replace v)
        } while (t != 0);
        return -u * (1L << k); // gcd is u*2^k
    }

    
    public static int lcm(int a, int b) throws MathArithmeticException {
        if (a == 0 || b == 0){
            return 0;
        }
        int lcm = FastMath.abs(ArithmeticUtils.mulAndCheck(a / gcd(a, b), b));
        if (lcm == Integer.MIN_VALUE) {
            throw new MathArithmeticException(LocalizedFormats.LCM_OVERFLOW_32_BITS,
                                              a, b);
        }
        return lcm;
    }

    
    public static long lcm(long a, long b) throws MathArithmeticException {
        if (a == 0 || b == 0){
            return 0;
        }
        long lcm = FastMath.abs(ArithmeticUtils.mulAndCheck(a / gcd(a, b), b));
        if (lcm == Long.MIN_VALUE){
            throw new MathArithmeticException(LocalizedFormats.LCM_OVERFLOW_64_BITS,
                                              a, b);
        }
        return lcm;
    }

    
    public static int mulAndCheck(int x, int y) throws MathArithmeticException {
        long m = ((long)x) * ((long)y);
        if (m < Integer.MIN_VALUE || m > Integer.MAX_VALUE) {
            throw new MathArithmeticException();
        }
        return (int)m;
    }

    
    public static long mulAndCheck(long a, long b) throws MathArithmeticException {
        long ret;
        if (a > b) {
            // use symmetry to reduce boundary cases
            ret = mulAndCheck(b, a);
        } else {
            if (a < 0) {
                if (b < 0) {
                    // check for positive overflow with negative a, negative b
                    if (a >= Long.MAX_VALUE / b) {
                        ret = a * b;
                    } else {
                        throw new MathArithmeticException();
                    }
                } else if (b > 0) {
                    // check for negative overflow with negative a, positive b
                    if (Long.MIN_VALUE / b <= a) {
                        ret = a * b;
                    } else {
                        throw new MathArithmeticException();

                    }
                } else {
                    // assert b == 0
                    ret = 0;
                }
            } else if (a > 0) {
                // assert a > 0
                // assert b > 0

                // check for positive overflow with positive a, positive b
                if (a <= Long.MAX_VALUE / b) {
                    ret = a * b;
                } else {
                    throw new MathArithmeticException();
                }
            } else {
                // assert a == 0
                ret = 0;
            }
        }
        return ret;
    }

    
    public static int subAndCheck(int x, int y) throws MathArithmeticException {
        long s = (long)x - (long)y;
        if (s < Integer.MIN_VALUE || s > Integer.MAX_VALUE) {
            throw new MathArithmeticException(LocalizedFormats.OVERFLOW_IN_SUBTRACTION, x, y);
        }
        return (int)s;
    }

    
    public static long subAndCheck(long a, long b) throws MathArithmeticException {
        long ret;
        if (b == Long.MIN_VALUE) {
            if (a < 0) {
                ret = a - b;
            } else {
                throw new MathArithmeticException(LocalizedFormats.OVERFLOW_IN_ADDITION, a, -b);
            }
        } else {
            // use additive inverse
            ret = addAndCheck(a, -b, LocalizedFormats.OVERFLOW_IN_ADDITION);
        }
        return ret;
    }

    
    public static int pow(final int k,
                          final int e)
        throws NotPositiveException,
               MathArithmeticException {
        if (e < 0) {
            throw new NotPositiveException(LocalizedFormats.EXPONENT, e);
        }

        try {
            int exp = e;
            int result = 1;
            int k2p    = k;
            while (true) {
                if ((exp & 0x1) != 0) {
                    result = mulAndCheck(result, k2p);
                }

                exp >>= 1;
                if (exp == 0) {
                    break;
                }

                k2p = mulAndCheck(k2p, k2p);
            }

            return result;
        } catch (MathArithmeticException mae) {
            // Add context information.
            mae.getContext().addMessage(LocalizedFormats.OVERFLOW);
            mae.getContext().addMessage(LocalizedFormats.BASE, k);
            mae.getContext().addMessage(LocalizedFormats.EXPONENT, e);

            // Rethrow.
            throw mae;
        }
    }

    
    @Deprecated
    public static int pow(final int k, long e) throws NotPositiveException {
        if (e < 0) {
            throw new NotPositiveException(LocalizedFormats.EXPONENT, e);
        }

        int result = 1;
        int k2p    = k;
        while (e != 0) {
            if ((e & 0x1) != 0) {
                result *= k2p;
            }
            k2p *= k2p;
            e >>= 1;
        }

        return result;
    }

    
    public static long pow(final long k,
                           final int e)
        throws NotPositiveException,
               MathArithmeticException {
        if (e < 0) {
            throw new NotPositiveException(LocalizedFormats.EXPONENT, e);
        }

        try {
            int exp = e;
            long result = 1;
            long k2p    = k;
            while (true) {
                if ((exp & 0x1) != 0) {
                    result = mulAndCheck(result, k2p);
                }

                exp >>= 1;
                if (exp == 0) {
                    break;
                }

                k2p = mulAndCheck(k2p, k2p);
            }

            return result;
        } catch (MathArithmeticException mae) {
            // Add context information.
            mae.getContext().addMessage(LocalizedFormats.OVERFLOW);
            mae.getContext().addMessage(LocalizedFormats.BASE, k);
            mae.getContext().addMessage(LocalizedFormats.EXPONENT, e);

            // Rethrow.
            throw mae;
        }
    }

    
    @Deprecated
    public static long pow(final long k, long e) throws NotPositiveException {
        if (e < 0) {
            throw new NotPositiveException(LocalizedFormats.EXPONENT, e);
        }

        long result = 1l;
        long k2p    = k;
        while (e != 0) {
            if ((e & 0x1) != 0) {
                result *= k2p;
            }
            k2p *= k2p;
            e >>= 1;
        }

        return result;
    }

    
    public static BigInteger pow(final BigInteger k, int e) throws NotPositiveException {
        if (e < 0) {
            throw new NotPositiveException(LocalizedFormats.EXPONENT, e);
        }

        return k.pow(e);
    }

    
    public static BigInteger pow(final BigInteger k, long e) throws NotPositiveException {
        if (e < 0) {
            throw new NotPositiveException(LocalizedFormats.EXPONENT, e);
        }

        BigInteger result = BigInteger.ONE;
        BigInteger k2p    = k;
        while (e != 0) {
            if ((e & 0x1) != 0) {
                result = result.multiply(k2p);
            }
            k2p = k2p.multiply(k2p);
            e >>= 1;
        }

        return result;

    }

    
    public static BigInteger pow(final BigInteger k, BigInteger e) throws NotPositiveException {
        if (e.compareTo(BigInteger.ZERO) < 0) {
            throw new NotPositiveException(LocalizedFormats.EXPONENT, e);
        }

        BigInteger result = BigInteger.ONE;
        BigInteger k2p    = k;
        while (!BigInteger.ZERO.equals(e)) {
            if (e.testBit(0)) {
                result = result.multiply(k2p);
            }
            k2p = k2p.multiply(k2p);
            e = e.shiftRight(1);
        }

        return result;
    }

    
    @Deprecated
    public static long stirlingS2(final int n, final int k)
        throws NotPositiveException, NumberIsTooLargeException, MathArithmeticException {
        return CombinatoricsUtils.stirlingS2(n, k);

    }

    
     private static long addAndCheck(long a, long b, Localizable pattern) throws MathArithmeticException {
         final long result = a + b;
         if (!((a ^ b) < 0 | (a ^ result) >= 0)) {
             throw new MathArithmeticException(pattern, a, b);
         }
         return result;
    }

    
    public static boolean isPowerOfTwo(long n) {
        return (n > 0) && ((n & (n - 1)) == 0);
    }
}
