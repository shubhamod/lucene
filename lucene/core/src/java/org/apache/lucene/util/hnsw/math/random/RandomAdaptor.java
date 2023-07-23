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


public class RandomAdaptor extends Random implements RandomGenerator {

    
    private static final long serialVersionUID = 2306581345647615033L;

    
    private final RandomGenerator randomGenerator;

    
    @SuppressWarnings("unused")
    private RandomAdaptor() { randomGenerator = null; }

    
    public RandomAdaptor(RandomGenerator randomGenerator) {
        this.randomGenerator = randomGenerator;
    }

    
    public static Random createAdaptor(RandomGenerator randomGenerator) {
        return new RandomAdaptor(randomGenerator);
    }

    
    @Override
    public boolean nextBoolean() {
        return randomGenerator.nextBoolean();
    }

     
    @Override
    public void nextBytes(byte[] bytes) {
        randomGenerator.nextBytes(bytes);
    }

     
    @Override
    public double nextDouble() {
        return randomGenerator.nextDouble();
    }

    
    @Override
    public float nextFloat() {
        return randomGenerator.nextFloat();
    }

    
    @Override
    public double nextGaussian() {
        return randomGenerator.nextGaussian();
    }

     
    @Override
    public int nextInt() {
        return randomGenerator.nextInt();
    }

    
    @Override
    public int nextInt(int n) {
        return randomGenerator.nextInt(n);
    }

    
    @Override
    public long nextLong() {
        return randomGenerator.nextLong();
    }

    
    public void setSeed(int seed) {
        if (randomGenerator != null) {  // required to avoid NPE in constructor
            randomGenerator.setSeed(seed);
        }
    }

    
    public void setSeed(int[] seed) {
        if (randomGenerator != null) {  // required to avoid NPE in constructor
            randomGenerator.setSeed(seed);
        }
    }

    
    @Override
    public void setSeed(long seed) {
        if (randomGenerator != null) {  // required to avoid NPE in constructor
            randomGenerator.setSeed(seed);
        }
    }

}
