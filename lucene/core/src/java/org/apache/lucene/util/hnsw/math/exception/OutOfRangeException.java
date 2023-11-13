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
package org.apache.lucene.util.hnsw.math.exception;

import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.exception.util.Localizable;


public class OutOfRangeException extends MathIllegalNumberException {
    
    private static final long serialVersionUID = 111601815794403609L;
    
    private final Number lo;
    
    private final Number hi;

    
    public OutOfRangeException(Number wrong,
                               Number lo,
                               Number hi) {
        this(LocalizedFormats.OUT_OF_RANGE_SIMPLE, wrong, lo, hi);
    }

    
    public OutOfRangeException(Localizable specific,
                               Number wrong,
                               Number lo,
                               Number hi) {
        super(specific, wrong, lo, hi);
        this.lo = lo;
        this.hi = hi;
    }

    
    public Number getLo() {
        return lo;
    }
    
    public Number getHi() {
        return hi;
    }
}
