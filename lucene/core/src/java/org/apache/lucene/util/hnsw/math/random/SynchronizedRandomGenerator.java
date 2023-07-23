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


public class SynchronizedRandomGenerator implements RandomGenerator {
    
    private final RandomGenerator wrapped;

    
    public SynchronizedRandomGenerator(RandomGenerator rng) {
        wrapped = rng;
    }

    
    public synchronized void setSeed(int seed) {
        wrapped.setSeed(seed);
    }

    
    public synchronized void setSeed(int[] seed) {
        wrapped.setSeed(seed);
    }

    
    public synchronized void setSeed(long seed) {
        wrapped.setSeed(seed);
    }

    
    public synchronized void nextBytes(byte[] bytes) {
        wrapped.nextBytes(bytes);
    }

    
    public synchronized int nextInt() {
        return wrapped.nextInt();
    }

    
    public synchronized int nextInt(int n) {
        return wrapped.nextInt(n);
    }

    
    public synchronized long nextLong() {
        return wrapped.nextLong();
    }

    
    public synchronized boolean nextBoolean() {
        return wrapped.nextBoolean();
    }

    
    public synchronized float nextFloat() {
        return wrapped.nextFloat();
    }

    
    public synchronized double nextDouble() {
        return wrapped.nextDouble();
    }

    
    public synchronized double nextGaussian() {
        return wrapped.nextGaussian();
    }
}
