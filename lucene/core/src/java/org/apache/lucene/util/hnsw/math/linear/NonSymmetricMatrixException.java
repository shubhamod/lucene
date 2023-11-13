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

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class NonSymmetricMatrixException extends MathIllegalArgumentException {
    
    private static final long serialVersionUID = -7518495577824189882L;
    
    private final int row;
    
    private final int column;
    
    private final double threshold;

    
    public NonSymmetricMatrixException(int row,
                                       int column,
                                       double threshold) {
        super(LocalizedFormats.NON_SYMMETRIC_MATRIX, row, column, threshold);
        this.row = row;
        this.column = column;
        this.threshold = threshold;
    }

    
    public int getRow() {
        return row;
    }
    
    public int getColumn() {
        return column;
    }
    
    public double getThreshold() {
        return threshold;
    }
}
