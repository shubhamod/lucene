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
package org.apache.lucene.util.hnsw.math.primes;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;

import java.util.List;



public class Primes {

    
    private Primes() {
    }

    
    public static boolean isPrime(int n) {
        if (n < 2) {
            return false;
        }

        for (int p : SmallPrimes.PRIMES) {
            if (0 == (n % p)) {
                return n == p;
            }
        }
        return SmallPrimes.millerRabinPrimeTest(n);
    }

    
    public static int nextPrime(int n) {
        if (n < 0) {
            throw new MathIllegalArgumentException(LocalizedFormats.NUMBER_TOO_SMALL, n, 0);
        }
        if (n == 2) {
            return 2;
        }
        n |= 1;//make sure n is odd
        if (n == 1) {
            return 2;
        }

        if (isPrime(n)) {
            return n;
        }

        // prepare entry in the +2, +4 loop:
        // n should not be a multiple of 3
        final int rem = n % 3;
        if (0 == rem) { // if n % 3 == 0
            n += 2; // n % 3 == 2
        } else if (1 == rem) { // if n % 3 == 1
            // if (isPrime(n)) return n;
            n += 4; // n % 3 == 2
        }
        while (true) { // this loop skips all multiple of 3
            if (isPrime(n)) {
                return n;
            }
            n += 2; // n % 3 == 1
            if (isPrime(n)) {
                return n;
            }
            n += 4; // n % 3 == 2
        }
    }

    
    public static List<Integer> primeFactors(int n) {

        if (n < 2) {
            throw new MathIllegalArgumentException(LocalizedFormats.NUMBER_TOO_SMALL, n, 2);
        }
        // slower than trial div unless we do an awful lot of computation
        // (then it finally gets JIT-compiled efficiently
        // List<Integer> out = PollardRho.primeFactors(n);
        return SmallPrimes.trialDivision(n);

    }

}
