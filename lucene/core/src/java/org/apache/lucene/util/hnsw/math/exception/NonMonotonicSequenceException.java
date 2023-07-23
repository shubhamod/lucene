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

import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class NonMonotonicSequenceException extends MathIllegalNumberException {
    
    private static final long serialVersionUID = 3596849179428944575L;
    
    private final MathArrays.OrderDirection direction;
    
    private final boolean strict;
    
    private final int index;
    
    private final Number previous;

    
    public NonMonotonicSequenceException(Number wrong,
                                         Number previous,
                                         int index) {
        this(wrong, previous, index, MathArrays.OrderDirection.INCREASING, true);
    }

    
    public NonMonotonicSequenceException(Number wrong,
                                         Number previous,
                                         int index,
                                         MathArrays.OrderDirection direction,
                                         boolean strict) {
        super(direction == MathArrays.OrderDirection.INCREASING ?
              (strict ?
               LocalizedFormats.NOT_STRICTLY_INCREASING_SEQUENCE :
               LocalizedFormats.NOT_INCREASING_SEQUENCE) :
              (strict ?
               LocalizedFormats.NOT_STRICTLY_DECREASING_SEQUENCE :
               LocalizedFormats.NOT_DECREASING_SEQUENCE),
              wrong, previous, Integer.valueOf(index), Integer.valueOf(index - 1));

        this.direction = direction;
        this.strict = strict;
        this.index = index;
        this.previous = previous;
    }

    
    public MathArrays.OrderDirection getDirection() {
        return direction;
    }
    
    public boolean getStrict() {
        return strict;
    }
    
    public int getIndex() {
        return index;
    }
    
    public Number getPrevious() {
        return previous;
    }
}
