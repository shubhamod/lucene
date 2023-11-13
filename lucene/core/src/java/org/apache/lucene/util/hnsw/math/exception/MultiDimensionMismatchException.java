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


public class MultiDimensionMismatchException extends MathIllegalArgumentException {
    
    private static final long serialVersionUID = -8415396756375798143L;

    
    private final Integer[] wrong;
    
    private final Integer[] expected;

    
    public MultiDimensionMismatchException(Integer[] wrong,
                                           Integer[] expected) {
        this(LocalizedFormats.DIMENSIONS_MISMATCH, wrong, expected);
    }

    
    public MultiDimensionMismatchException(Localizable specific,
                                           Integer[] wrong,
                                           Integer[] expected) {
        super(specific, wrong, expected);
        this.wrong = wrong.clone();
        this.expected = expected.clone();
    }

    
    public Integer[] getWrongDimensions() {
        return wrong.clone();
    }
    
    public Integer[] getExpectedDimensions() {
        return expected.clone();
    }

    
    public int getWrongDimension(int index) {
        return wrong[index].intValue();
    }
    
    public int getExpectedDimension(int index) {
        return expected[index].intValue();
    }
}
