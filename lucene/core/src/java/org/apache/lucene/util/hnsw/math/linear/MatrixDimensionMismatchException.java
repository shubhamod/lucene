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

import org.apache.lucene.util.hnsw.math.exception.MultiDimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class MatrixDimensionMismatchException extends MultiDimensionMismatchException {
    
    private static final long serialVersionUID = -8415396756375798143L;

    
    public MatrixDimensionMismatchException(int wrongRowDim,
                                            int wrongColDim,
                                            int expectedRowDim,
                                            int expectedColDim) {
        super(LocalizedFormats.DIMENSIONS_MISMATCH_2x2,
              new Integer[] { wrongRowDim, wrongColDim },
              new Integer[] { expectedRowDim, expectedColDim });
    }

    
    public int getWrongRowDimension() {
        return getWrongDimension(0);
    }
    
    public int getExpectedRowDimension() {
        return getExpectedDimension(0);
    }
    
    public int getWrongColumnDimension() {
        return getWrongDimension(1);
    }
    
    public int getExpectedColumnDimension() {
        return getExpectedDimension(1);
    }
}
