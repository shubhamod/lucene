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


public class NumberIsTooLargeException extends MathIllegalNumberException {
    
    private static final long serialVersionUID = 4330003017885151975L;
    
    private final Number max;
    
    private final boolean boundIsAllowed;

    
    public NumberIsTooLargeException(Number wrong,
                                     Number max,
                                     boolean boundIsAllowed) {
        this(boundIsAllowed ?
             LocalizedFormats.NUMBER_TOO_LARGE :
             LocalizedFormats.NUMBER_TOO_LARGE_BOUND_EXCLUDED,
             wrong, max, boundIsAllowed);
    }
    
    public NumberIsTooLargeException(Localizable specific,
                                     Number wrong,
                                     Number max,
                                     boolean boundIsAllowed) {
        super(specific, wrong, max);

        this.max = max;
        this.boundIsAllowed = boundIsAllowed;
    }

    
    public boolean getBoundIsAllowed() {
        return boundIsAllowed;
    }

    
    public Number getMax() {
        return max;
    }
}
