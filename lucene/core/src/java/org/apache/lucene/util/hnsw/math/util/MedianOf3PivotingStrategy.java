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

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;



public class MedianOf3PivotingStrategy implements PivotingStrategyInterface, Serializable {

    
    private static final long serialVersionUID = 20140713L;

    
    public int pivotIndex(final double[] work, final int begin, final int end)
        throws MathIllegalArgumentException {
        MathArrays.verifyValues(work, begin, end-begin);
        final int inclusiveEnd = end - 1;
        final int middle = begin + (inclusiveEnd - begin) / 2;
        final double wBegin = work[begin];
        final double wMiddle = work[middle];
        final double wEnd = work[inclusiveEnd];

        if (wBegin < wMiddle) {
            if (wMiddle < wEnd) {
                return middle;
            } else {
                return wBegin < wEnd ? inclusiveEnd : begin;
            }
        } else {
            if (wBegin < wEnd) {
                return begin;
            } else {
                return wMiddle < wEnd ? inclusiveEnd : middle;
            }
        }
    }

}
