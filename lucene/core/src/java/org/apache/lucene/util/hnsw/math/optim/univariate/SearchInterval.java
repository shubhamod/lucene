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
package org.apache.lucene.util.hnsw.math.optim.univariate;

import org.apache.lucene.util.hnsw.math.optim.OptimizationData;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;


public class SearchInterval implements OptimizationData {
    
    private final double lower;
    
    private final double upper;
    
    private final double start;

    
    public SearchInterval(double lo,
                          double hi,
                          double init) {
        if (lo >= hi) {
            throw new NumberIsTooLargeException(lo, hi, false);
        }
        if (init < lo ||
            init > hi) {
            throw new OutOfRangeException(init, lo, hi);
        }

        lower = lo;
        upper = hi;
        start = init;
    }

    
    public SearchInterval(double lo,
                          double hi) {
        this(lo, hi, 0.5 * (lo + hi));
    }

    
    public double getMin() {
        return lower;
    }
    
    public double getMax() {
        return upper;
    }
    
    public double getStartValue() {
        return start;
    }
}
