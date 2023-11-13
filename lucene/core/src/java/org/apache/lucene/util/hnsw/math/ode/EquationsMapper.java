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

package org.apache.lucene.util.hnsw.math.ode;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;


public class EquationsMapper implements Serializable {

    
    private static final long serialVersionUID = 20110925L;

    
    private final int firstIndex;

    
    private final int dimension;

    
    public EquationsMapper(final int firstIndex, final int dimension) {
        this.firstIndex = firstIndex;
        this.dimension  = dimension;
    }

    
    public int getFirstIndex() {
        return firstIndex;
    }

    
    public int getDimension() {
        return dimension;
    }

    
    public void extractEquationData(double[] complete, double[] equationData)
        throws DimensionMismatchException {
        if (equationData.length != dimension) {
            throw new DimensionMismatchException(equationData.length, dimension);
        }
        System.arraycopy(complete, firstIndex, equationData, 0, dimension);
    }

    
    public void insertEquationData(double[] equationData, double[] complete)
        throws DimensionMismatchException {
        if (equationData.length != dimension) {
            throw new DimensionMismatchException(equationData.length, dimension);
        }
        System.arraycopy(equationData, 0, complete, firstIndex, dimension);
    }

}
