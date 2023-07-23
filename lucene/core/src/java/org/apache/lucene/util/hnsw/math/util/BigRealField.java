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

package org.apache.lucene.util.hnsw.math.util;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;


public class BigRealField implements Field<BigReal>, Serializable  {

    
    private static final long serialVersionUID = 4756431066541037559L;

    
    private BigRealField() {
    }

    
    public static BigRealField getInstance() {
        return LazyHolder.INSTANCE;
    }

    
    public BigReal getOne() {
        return BigReal.ONE;
    }

    
    public BigReal getZero() {
        return BigReal.ZERO;
    }

    
    public Class<? extends FieldElement<BigReal>> getRuntimeClass() {
        return BigReal.class;
    }

    // CHECKSTYLE: stop HideUtilityClassConstructor
    
    private static class LazyHolder {
        
        private static final BigRealField INSTANCE = new BigRealField();
    }
    // CHECKSTYLE: resume HideUtilityClassConstructor

    
    private Object readResolve() {
        // return the singleton instance
        return LazyHolder.INSTANCE;
    }

}
