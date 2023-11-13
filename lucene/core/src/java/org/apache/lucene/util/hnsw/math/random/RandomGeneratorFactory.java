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

import java.util.Random;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;


public class RandomGeneratorFactory {
    
    private RandomGeneratorFactory() {}

    
    public static RandomGenerator createRandomGenerator(final Random rng) {
        return new RandomGenerator() {
            
            public void setSeed(int seed) {
                rng.setSeed((long) seed);
            }

            
            public void setSeed(int[] seed) {
                rng.setSeed(convertToLong(seed));
            }

            
            public void setSeed(long seed) {
                rng.setSeed(seed);
            }

            
            public void nextBytes(byte[] bytes) {
                rng.nextBytes(bytes);
            }

            
            public int nextInt() {
                return rng.nextInt();
            }

            
            public int nextInt(int n) {
                if (n <= 0) {
                    throw new NotStrictlyPositiveException(n);
                }
                return rng.nextInt(n);
            }

            
            public long nextLong() {
                return rng.nextLong();
            }

            
            public boolean nextBoolean() {
                return rng.nextBoolean();
            }

            
            public float nextFloat() {
                return rng.nextFloat();
            }

            
            public double nextDouble() {
                return rng.nextDouble();
            }

            
            public double nextGaussian() {
                return rng.nextGaussian();
            }
        };
    }

    
    public static long convertToLong(int[] seed) {
        // The following number is the largest prime that fits
        // in 32 bits (i.e. 2^32 - 5).
        final long prime = 4294967291l;

        long combined = 0l;
        for (int s : seed) {
            combined = combined * prime + s;
        }

        return combined;
    }
}
