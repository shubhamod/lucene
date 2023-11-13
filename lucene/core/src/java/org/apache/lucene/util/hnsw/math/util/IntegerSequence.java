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
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathUnsupportedOperationException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;


public class IntegerSequence {
    
    private IntegerSequence() {}

    
    public static Range range(int start,
                              int end) {
        return range(start, end, 1);
    }

    
    public static Range range(final int start,
                              final int max,
                              final int step) {
        return new Range(start, max, step);
    }

    
    public static class Range implements Iterable<Integer> {
        
        private final int size;
        
        private final int start;
        
        private final int max;
        
        private final int step;

        
        public Range(int start,
                     int max,
                     int step) {
            this.start = start;
            this.max = max;
            this.step = step;

            final int s = (max - start) / step + 1;
            this.size = s < 0 ? 0 : s;
        }

        
        public int size() {
            return size;
        }

        
        public Iterator<Integer> iterator() {
            return Incrementor.create()
                .withStart(start)
                .withMaximalCount(max + (step > 0 ? 1 : -1))
                .withIncrement(step);
        }
    }

    
    public static class Incrementor implements Iterator<Integer> {
        
        private static final MaxCountExceededCallback CALLBACK
            = new MaxCountExceededCallback() {
                    
                    public void trigger(int max) throws MaxCountExceededException {
                        throw new MaxCountExceededException(max);
                    }
                };

        
        private final int init;
        
        private final int maximalCount;
        
        private final int increment;
        
        private final MaxCountExceededCallback maxCountCallback;
        
        private int count = 0;

        
        public interface MaxCountExceededCallback {
            
            void trigger(int maximalCount) throws MaxCountExceededException;
        }

        
        private Incrementor(int start,
                            int max,
                            int step,
                            MaxCountExceededCallback cb)
            throws NullArgumentException {
            if (cb == null) {
                throw new NullArgumentException();
            }
            this.init = start;
            this.maximalCount = max;
            this.increment = step;
            this.maxCountCallback = cb;
            this.count = start;
        }

        
        public static Incrementor create() {
            return new Incrementor(0, 0, 1, CALLBACK);
        }

        
        public Incrementor withStart(int start) {
            return new Incrementor(start,
                                   this.maximalCount,
                                   this.increment,
                                   this.maxCountCallback);
        }

        
        public Incrementor withMaximalCount(int max) {
            return new Incrementor(this.init,
                                   max,
                                   this.increment,
                                   this.maxCountCallback);
        }

        
        public Incrementor withIncrement(int step) {
            if (step == 0) {
                throw new ZeroException();
            }
            return new Incrementor(this.init,
                                   this.maximalCount,
                                   step,
                                   this.maxCountCallback);
        }

        
        public Incrementor withCallback(MaxCountExceededCallback cb) {
            return new Incrementor(this.init,
                                   this.maximalCount,
                                   this.increment,
                                   cb);
        }

        
        public int getMaximalCount() {
            return maximalCount;
        }

        
        public int getCount() {
            return count;
        }

        
        public boolean canIncrement() {
            return canIncrement(1);
        }

        
        public boolean canIncrement(int nTimes) {
            final int finalCount = count + nTimes * increment;
            return increment < 0 ?
                finalCount > maximalCount :
                finalCount < maximalCount;
        }

        
        public void increment(int nTimes) throws MaxCountExceededException {
            if (nTimes <= 0) {
                throw new NotStrictlyPositiveException(nTimes);
            }

            if (!canIncrement(0)) {
                maxCountCallback.trigger(maximalCount);
            }
            count += nTimes * increment;
        }

        
        public void increment() throws MaxCountExceededException {
            increment(1);
        }

        
        public boolean hasNext() {
            return canIncrement(0);
        }

        
        public Integer next() {
            final int value = count;
            increment();
            return value;
        }

        
        public void remove() {
            throw new MathUnsupportedOperationException();
        }
    }
}
