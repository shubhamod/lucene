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

import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public abstract class AbstractRandomGenerator implements RandomGenerator {

    
    private double cachedNormalDeviate = Double.NaN;

    
    public AbstractRandomGenerator() {
        super();

    }

    
    public void clear() {
        cachedNormalDeviate = Double.NaN;
    }

    
    public void setSeed(int seed) {
        setSeed((long) seed);
    }

    
    public void setSeed(int[] seed) {
        // the following number is the largest prime that fits in 32 bits (it is 2^32 - 5)
        final long prime = 4294967291l;

        long combined = 0l;
        for (int s : seed) {
            combined = combined * prime + s;
        }
        setSeed(combined);
    }

    
    public abstract void setSeed(long seed);

    
    public void nextBytes(byte[] bytes) {
        int bytesOut = 0;
        while (bytesOut < bytes.length) {
          int randInt = nextInt();
          for (int i = 0; i < 3; i++) {
              if ( i > 0) {
                  randInt >>= 8;
              }
              bytes[bytesOut++] = (byte) randInt;
              if (bytesOut == bytes.length) {
                  return;
              }
          }
        }
    }

     
    public int nextInt() {
        return (int) ((2d * nextDouble() - 1d) * Integer.MAX_VALUE);
    }

    
    public int nextInt(int n) {
        if (n <= 0 ) {
            throw new NotStrictlyPositiveException(n);
        }
        int result = (int) (nextDouble() * n);
        return result < n ? result : n - 1;
    }

     
    public long nextLong() {
        return (long) ((2d * nextDouble() - 1d) * Long.MAX_VALUE);
    }

    
    public boolean nextBoolean() {
        return nextDouble() <= 0.5;
    }

     
    public float nextFloat() {
        return (float) nextDouble();
    }

    
    public abstract double nextDouble();

    
    public double nextGaussian() {
        if (!Double.isNaN(cachedNormalDeviate)) {
            double dev = cachedNormalDeviate;
            cachedNormalDeviate = Double.NaN;
            return dev;
        }
        double v1 = 0;
        double v2 = 0;
        double s = 1;
        while (s >=1 ) {
            v1 = 2 * nextDouble() - 1;
            v2 = 2 * nextDouble() - 1;
            s = v1 * v1 + v2 * v2;
        }
        if (s != 0) {
            s = FastMath.sqrt(-2 * FastMath.log(s) / s);
        }
        cachedNormalDeviate = v2 * s;
        return v1 * s;
    }
}
