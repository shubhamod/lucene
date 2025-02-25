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

import org.apache.lucene.util.hnsw.math.util.FastMath;



public class MersenneTwister extends BitsStreamGenerator implements Serializable {

    
    private static final long serialVersionUID = 8661194735290153518L;

    
    private static final int   N     = 624;

    
    private static final int   M     = 397;

    
    private static final int[] MAG01 = { 0x0, 0x9908b0df };

    
    private int[] mt;

    
    private int   mti;

    
    public MersenneTwister() {
        mt = new int[N];
        setSeed(System.currentTimeMillis() + System.identityHashCode(this));
    }

    
    public MersenneTwister(int seed) {
        mt = new int[N];
        setSeed(seed);
    }

    
    public MersenneTwister(int[] seed) {
        mt = new int[N];
        setSeed(seed);
    }

    
    public MersenneTwister(long seed) {
        mt = new int[N];
        setSeed(seed);
    }

    
    @Override
    public void setSeed(int seed) {
        // we use a long masked by 0xffffffffL as a poor man unsigned int
        long longMT = seed;
        // NB: unlike original C code, we are working with java longs, the cast below makes masking unnecessary
        mt[0]= (int) longMT;
        for (mti = 1; mti < N; ++mti) {
            // See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier.
            // initializer from the 2002-01-09 C version by Makoto Matsumoto
            longMT = (1812433253l * (longMT ^ (longMT >> 30)) + mti) & 0xffffffffL;
            mt[mti]= (int) longMT;
        }

        clear(); // Clear normal deviate cache
    }

    
    @Override
    public void setSeed(int[] seed) {

        if (seed == null) {
            setSeed(System.currentTimeMillis() + System.identityHashCode(this));
            return;
        }

        setSeed(19650218);
        int i = 1;
        int j = 0;

        for (int k = FastMath.max(N, seed.length); k != 0; k--) {
            long l0 = (mt[i] & 0x7fffffffl)   | ((mt[i]   < 0) ? 0x80000000l : 0x0l);
            long l1 = (mt[i-1] & 0x7fffffffl) | ((mt[i-1] < 0) ? 0x80000000l : 0x0l);
            long l  = (l0 ^ ((l1 ^ (l1 >> 30)) * 1664525l)) + seed[j] + j; // non linear
            mt[i]   = (int) (l & 0xffffffffl);
            i++; j++;
            if (i >= N) {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if (j >= seed.length) {
                j = 0;
            }
        }

        for (int k = N - 1; k != 0; k--) {
            long l0 = (mt[i] & 0x7fffffffl)   | ((mt[i]   < 0) ? 0x80000000l : 0x0l);
            long l1 = (mt[i-1] & 0x7fffffffl) | ((mt[i-1] < 0) ? 0x80000000l : 0x0l);
            long l  = (l0 ^ ((l1 ^ (l1 >> 30)) * 1566083941l)) - i; // non linear
            mt[i]   = (int) (l & 0xffffffffL);
            i++;
            if (i >= N) {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }

        mt[0] = 0x80000000; // MSB is 1; assuring non-zero initial array

        clear(); // Clear normal deviate cache

    }

    
    @Override
    public void setSeed(long seed) {
        setSeed(new int[] { (int) (seed >>> 32), (int) (seed & 0xffffffffl) });
    }

    
    @Override
    protected int next(int bits) {

        int y;

        if (mti >= N) { // generate N words at one time
            int mtNext = mt[0];
            for (int k = 0; k < N - M; ++k) {
                int mtCurr = mtNext;
                mtNext = mt[k + 1];
                y = (mtCurr & 0x80000000) | (mtNext & 0x7fffffff);
                mt[k] = mt[k + M] ^ (y >>> 1) ^ MAG01[y & 0x1];
            }
            for (int k = N - M; k < N - 1; ++k) {
                int mtCurr = mtNext;
                mtNext = mt[k + 1];
                y = (mtCurr & 0x80000000) | (mtNext & 0x7fffffff);
                mt[k] = mt[k + (M - N)] ^ (y >>> 1) ^ MAG01[y & 0x1];
            }
            y = (mtNext & 0x80000000) | (mt[0] & 0x7fffffff);
            mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ MAG01[y & 0x1];

            mti = 0;
        }

        y = mt[mti++];

        // tempering
        y ^=  y >>> 11;
        y ^= (y <<   7) & 0x9d2c5680;
        y ^= (y <<  15) & 0xefc60000;
        y ^=  y >>> 18;

        return y >>> (32 - bits);

    }

}
