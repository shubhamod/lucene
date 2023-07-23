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

import java.util.NoSuchElementException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;


public class MultidimensionalCounter implements Iterable<Integer> {
    
    private final int dimension;
    
    private final int[] uniCounterOffset;
    
    private final int[] size;
    
    private final int totalSize;
    
    private final int last;

    
    public class Iterator implements java.util.Iterator<Integer> {
        
        private final int[] counter = new int[dimension];
        
        private int count = -1;
        
        private final int maxCount = totalSize - 1;

        
        Iterator() {
            counter[last] = -1;
        }

        
        public boolean hasNext() {
            return count < maxCount;
        }

        
        public Integer next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }

            for (int i = last; i >= 0; i--) {
                if (counter[i] == size[i] - 1) {
                    counter[i] = 0;
                } else {
                    ++counter[i];
                    break;
                }
            }

            return ++count;
        }

        
        public int getCount() {
            return count;
        }
        
        public int[] getCounts() {
            return MathArrays.copyOf(counter);
        }

        
        public int getCount(int dim) {
            return counter[dim];
        }

        
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    
    public MultidimensionalCounter(int ... size) throws NotStrictlyPositiveException {
        dimension = size.length;
        this.size = MathArrays.copyOf(size);

        uniCounterOffset = new int[dimension];

        last = dimension - 1;
        int tS = size[last];
        for (int i = 0; i < last; i++) {
            int count = 1;
            for (int j = i + 1; j < dimension; j++) {
                count *= size[j];
            }
            uniCounterOffset[i] = count;
            tS *= size[i];
        }
        uniCounterOffset[last] = 0;

        if (tS <= 0) {
            throw new NotStrictlyPositiveException(tS);
        }

        totalSize = tS;
    }

    
    public Iterator iterator() {
        return new Iterator();
    }

    
    public int getDimension() {
        return dimension;
    }

    
    public int[] getCounts(int index) throws OutOfRangeException {
        if (index < 0 ||
            index >= totalSize) {
            throw new OutOfRangeException(index, 0, totalSize);
        }

        final int[] indices = new int[dimension];

        int count = 0;
        for (int i = 0; i < last; i++) {
            int idx = 0;
            final int offset = uniCounterOffset[i];
            while (count <= index) {
                count += offset;
                ++idx;
            }
            --idx;
            count -= offset;
            indices[i] = idx;
        }

        indices[last] = index - count;

        return indices;
    }

    
    public int getCount(int ... c)
        throws OutOfRangeException, DimensionMismatchException {
        if (c.length != dimension) {
            throw new DimensionMismatchException(c.length, dimension);
        }
        int count = 0;
        for (int i = 0; i < dimension; i++) {
            final int index = c[i];
            if (index < 0 ||
                index >= size[i]) {
                throw new OutOfRangeException(index, 0, size[i] - 1);
            }
            count += uniCounterOffset[i] * c[i];
        }
        return count + c[last];
    }

    
    public int getSize() {
        return totalSize;
    }
    
    public int[] getSizes() {
        return MathArrays.copyOf(size);
    }

    
    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        for (int i = 0; i < dimension; i++) {
            sb.append("[").append(getCount(i)).append("]");
        }
        return sb.toString();
    }
}
