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
package org.apache.lucene.util.hnsw.math.linear;

import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.exception.util.ExceptionContext;


public class NonPositiveDefiniteMatrixException extends NumberIsTooSmallException {
    
    private static final long serialVersionUID = 1641613838113738061L;
    
    private final int index;
    
    private final double threshold;

    
    public NonPositiveDefiniteMatrixException(double wrong,
                                              int index,
                                              double threshold) {
        super(wrong, threshold, false);
        this.index = index;
        this.threshold = threshold;

        final ExceptionContext context = getContext();
        context.addMessage(LocalizedFormats.NOT_POSITIVE_DEFINITE_MATRIX);
        context.addMessage(LocalizedFormats.ARRAY_ELEMENT, wrong, index);
    }

    
    public int getRow() {
        return index;
    }
    
    public int getColumn() {
        return index;
    }
    
    public double getThreshold() {
        return threshold;
    }
}
