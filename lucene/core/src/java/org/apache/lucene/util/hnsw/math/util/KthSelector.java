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

import java.io.Serializable;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;



public class KthSelector implements Serializable {

    
    private static final long serialVersionUID = 20140713L;

    
    private static final int MIN_SELECT_SIZE = 15;

    
    private final PivotingStrategyInterface pivotingStrategy;

    
    public KthSelector() {
        this.pivotingStrategy = new MedianOf3PivotingStrategy();
    }

    
    public KthSelector(final PivotingStrategyInterface pivotingStrategy)
        throws NullArgumentException {
        MathUtils.checkNotNull(pivotingStrategy);
        this.pivotingStrategy = pivotingStrategy;
    }

    
    public PivotingStrategyInterface getPivotingStrategy() {
        return pivotingStrategy;
    }

    
    public double select(final double[] work, final int[] pivotsHeap, final int k) {
        int begin = 0;
        int end = work.length;
        int node = 0;
        final boolean usePivotsHeap = pivotsHeap != null;
        while (end - begin > MIN_SELECT_SIZE) {
            final int pivot;

            if (usePivotsHeap && node < pivotsHeap.length &&
                    pivotsHeap[node] >= 0) {
                // the pivot has already been found in a previous call
                // and the array has already been partitioned around it
                pivot = pivotsHeap[node];
            } else {
                // select a pivot and partition work array around it
                pivot = partition(work, begin, end, pivotingStrategy.pivotIndex(work, begin, end));
                if (usePivotsHeap && node < pivotsHeap.length) {
                    pivotsHeap[node] = pivot;
                }
            }

            if (k == pivot) {
                // the pivot was exactly the element we wanted
                return work[k];
            } else if (k < pivot) {
                // the element is in the left partition
                end  = pivot;
                node = FastMath.min(2 * node + 1, usePivotsHeap ? pivotsHeap.length : end);
            } else {
                // the element is in the right partition
                begin = pivot + 1;
                node  = FastMath.min(2 * node + 2, usePivotsHeap ? pivotsHeap.length : end);
            }
        }
        Arrays.sort(work, begin, end);
        return work[k];
    }

    
    private int partition(final double[] work, final int begin, final int end, final int pivot) {

        final double value = work[pivot];
        work[pivot] = work[begin];

        int i = begin + 1;
        int j = end - 1;
        while (i < j) {
            while (i < j && work[j] > value) {
                --j;
            }
            while (i < j && work[i] < value) {
                ++i;
            }

            if (i < j) {
                final double tmp = work[i];
                work[i++] = work[j];
                work[j--] = tmp;
            }
        }

        if (i >= end || work[i] > value) {
            --i;
        }
        work[begin] = work[i];
        work[i] = value;
        return i;
    }
}
