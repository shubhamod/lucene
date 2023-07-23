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

import java.util.Iterator;
import java.util.Comparator;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.io.Serializable;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;


public class Combinations implements Iterable<int[]> {
    
    private final int n;
    
    private final int k;
    
    private final IterationOrder iterationOrder;

    
    private enum IterationOrder {
        
        LEXICOGRAPHIC
    }

   
    public Combinations(int n,
                        int k) {
        this(n, k, IterationOrder.LEXICOGRAPHIC);
    }

    
    private Combinations(int n,
                         int k,
                         IterationOrder iterationOrder) {
        CombinatoricsUtils.checkBinomial(n, k);
        this.n = n;
        this.k = k;
        this.iterationOrder = iterationOrder;
    }

    
    public int getN() {
        return n;
    }

    
    public int getK() {
        return k;
    }

    
    public Iterator<int[]> iterator() {
        if (k == 0 ||
            k == n) {
            return new SingletonIterator(MathArrays.natural(k));
        }

        switch (iterationOrder) {
        case LEXICOGRAPHIC:
            return new LexicographicIterator(n, k);
        default:
            throw new MathInternalError(); // Should never happen.
        }
    }

    
    public Comparator<int[]> comparator() {
        return new LexicographicComparator(n, k);
    }

    
    private static class LexicographicIterator implements Iterator<int[]> {
        
        private final int k;

        
        private final int[] c;

        
        private boolean more = true;

        
        private int j;

        
        LexicographicIterator(int n, int k) {
            this.k = k;
            c = new int[k + 3];
            if (k == 0 || k >= n) {
                more = false;
                return;
            }
            // Initialize c to start with lexicographically first k-set
            for (int i = 1; i <= k; i++) {
                c[i] = i - 1;
            }
            // Initialize sentinels
            c[k + 1] = n;
            c[k + 2] = 0;
            j = k; // Set up invariant: j is smallest index such that c[j + 1] > j
        }

        
        public boolean hasNext() {
            return more;
        }

        
        public int[] next() {
            if (!more) {
                throw new NoSuchElementException();
            }
            // Copy return value (prepared by last activation)
            final int[] ret = new int[k];
            System.arraycopy(c, 1, ret, 0, k);

            // Prepare next iteration
            // T2 and T6 loop
            int x = 0;
            if (j > 0) {
                x = j;
                c[j] = x;
                j--;
                return ret;
            }
            // T3
            if (c[1] + 1 < c[2]) {
                c[1]++;
                return ret;
            } else {
                j = 2;
            }
            // T4
            boolean stepDone = false;
            while (!stepDone) {
                c[j - 1] = j - 2;
                x = c[j] + 1;
                if (x == c[j + 1]) {
                    j++;
                } else {
                    stepDone = true;
                }
            }
            // T5
            if (j > k) {
                more = false;
                return ret;
            }
            // T6
            c[j] = x;
            j--;
            return ret;
        }

        
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    
    private static class SingletonIterator implements Iterator<int[]> {
        
        private final int[] singleton;
        
        private boolean more = true;
        
        SingletonIterator(final int[] singleton) {
            this.singleton = singleton;
        }
        
        public boolean hasNext() {
            return more;
        }
        
        public int[] next() {
            if (more) {
                more = false;
                return singleton;
            } else {
                throw new NoSuchElementException();
            }
        }
        
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    
    private static class LexicographicComparator
        implements Comparator<int[]>, Serializable {
        
        private static final long serialVersionUID = 20130906L;
        
        private final int n;
        
        private final int k;

        
        LexicographicComparator(int n, int k) {
            this.n = n;
            this.k = k;
        }

        
        public int compare(int[] c1,
                           int[] c2) {
            if (c1.length != k) {
                throw new DimensionMismatchException(c1.length, k);
            }
            if (c2.length != k) {
                throw new DimensionMismatchException(c2.length, k);
            }

            // Method "lexNorm" works with ordered arrays.
            final int[] c1s = MathArrays.copyOf(c1);
            Arrays.sort(c1s);
            final int[] c2s = MathArrays.copyOf(c2);
            Arrays.sort(c2s);

            final long v1 = lexNorm(c1s);
            final long v2 = lexNorm(c2s);

            if (v1 < v2) {
                return -1;
            } else if (v1 > v2) {
                return 1;
            } else {
                return 0;
            }
        }

        
        private long lexNorm(int[] c) {
            long ret = 0;
            for (int i = 0; i < c.length; i++) {
                final int digit = c[i];
                if (digit < 0 ||
                    digit >= n) {
                    throw new OutOfRangeException(digit, 0, n - 1);
                }

                ret += c[i] * ArithmeticUtils.pow(n, i);
            }
            return ret;
        }
    }
}
