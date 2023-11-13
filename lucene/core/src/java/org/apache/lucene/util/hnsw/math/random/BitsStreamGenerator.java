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
package org.apache.lucene.util.hnsw.math.random;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public abstract class BitsStreamGenerator
    implements RandomGenerator,
               Serializable {
    
    private static final long serialVersionUID = 20130104L;
    
    private double nextGaussian;

    
    public BitsStreamGenerator() {
        nextGaussian = Double.NaN;
    }

    
    public abstract void setSeed(int seed);

    
    public abstract void setSeed(int[] seed);

    
    public abstract void setSeed(long seed);

    
    protected abstract int next(int bits);

    
    public boolean nextBoolean() {
        return next(1) != 0;
    }

    
    public double nextDouble() {
        final long high = ((long) next(26)) << 26;
        final int  low  = next(26);
        return (high | low) * 0x1.0p-52d;
    }

    
    public float nextFloat() {
        return next(23) * 0x1.0p-23f;
    }

    
    public double nextGaussian() {

        final double random;
        if (Double.isNaN(nextGaussian)) {
            // generate a new pair of gaussian numbers
            final double x = nextDouble();
            final double y = nextDouble();
            final double alpha = 2 * FastMath.PI * x;
            final double r      = FastMath.sqrt(-2 * FastMath.log(y));
            random       = r * FastMath.cos(alpha);
            nextGaussian = r * FastMath.sin(alpha);
        } else {
            // use the second element of the pair already generated
            random = nextGaussian;
            nextGaussian = Double.NaN;
        }

        return random;

    }

    
    public int nextInt() {
        return next(32);
    }

    
    public int nextInt(int n) throws IllegalArgumentException {
        if (n > 0) {
            if ((n & -n) == n) {
                return (int) ((n * (long) next(31)) >> 31);
            }
            int bits;
            int val;
            do {
                bits = next(31);
                val = bits % n;
            } while (bits - val + (n - 1) < 0);
            return val;
        }
        throw new NotStrictlyPositiveException(n);
    }

    
    public long nextLong() {
        final long high  = ((long) next(32)) << 32;
        final long  low  = ((long) next(32)) & 0xffffffffL;
        return high | low;
    }

    
    public long nextLong(long n) throws IllegalArgumentException {
        if (n > 0) {
            long bits;
            long val;
            do {
                bits = ((long) next(31)) << 32;
                bits |= ((long) next(32)) & 0xffffffffL;
                val  = bits % n;
            } while (bits - val + (n - 1) < 0);
            return val;
        }
        throw new NotStrictlyPositiveException(n);
    }

    
    public void clear() {
        nextGaussian = Double.NaN;
    }

    
    public void nextBytes(byte[] bytes) {
        nextBytesFill(bytes, 0, bytes.length);
    }

    
    public void nextBytes(byte[] bytes,
                          int start,
                          int len) {
        if (start < 0 ||
            start >= bytes.length) {
            throw new OutOfRangeException(start, 0, bytes.length);
        }
        if (len < 0 ||
            len > bytes.length - start) {
            throw new OutOfRangeException(len, 0, bytes.length - start);
        }

        nextBytesFill(bytes, start, len);
    }

    
    private void nextBytesFill(byte[] bytes,
                               int start,
                               int len) {
        int index = start; // Index of first insertion.

        // Index of first insertion plus multiple 4 part of length (i.e. length
        // with two least significant bits unset).
        final int indexLoopLimit = index + (len & 0x7ffffffc);

        // Start filling in the byte array, 4 bytes at a time.
        while (index < indexLoopLimit) {
            final int random = next(32);
            bytes[index++] = (byte) random;
            bytes[index++] = (byte) (random >>> 8);
            bytes[index++] = (byte) (random >>> 16);
            bytes[index++] = (byte) (random >>> 24);
        }

        final int indexLimit = start + len; // Index of last insertion + 1.

        // Fill in the remaining bytes.
        if (index < indexLimit) {
            int random = next(32);
            while (true) {
                bytes[index++] = (byte) random;
                if (index < indexLimit) {
                    random >>>= 8;
                } else {
                    break;
                }
            }
        }
    }
}
