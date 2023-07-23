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

import org.apache.lucene.util.hnsw.math.exception.util.Localizable;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class NumberIsTooSmallException extends MathIllegalNumberException {
    
    private static final long serialVersionUID = -6100997100383932834L;
    
    private final Number min;
    
    private final boolean boundIsAllowed;

    
    public NumberIsTooSmallException(Number wrong,
                                     Number min,
                                     boolean boundIsAllowed) {
        this(boundIsAllowed ?
             LocalizedFormats.NUMBER_TOO_SMALL :
             LocalizedFormats.NUMBER_TOO_SMALL_BOUND_EXCLUDED,
             wrong, min, boundIsAllowed);
    }

    
    public NumberIsTooSmallException(Localizable specific,
                                     Number wrong,
                                     Number min,
                                     boolean boundIsAllowed) {
        super(specific, wrong, min);

        this.min = min;
        this.boundIsAllowed = boundIsAllowed;
    }

    
    public boolean getBoundIsAllowed() {
        return boundIsAllowed;
    }

    
    public Number getMin() {
        return min;
    }
}
