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

package org.apache.lucene.util.hnsw.math.fraction;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;


public class FractionField implements Field<Fraction>, Serializable  {

    
    private static final long serialVersionUID = -1257768487499119313L;

    
    private FractionField() {
    }

    
    public static FractionField getInstance() {
        return LazyHolder.INSTANCE;
    }

    
    public Fraction getOne() {
        return Fraction.ONE;
    }

    
    public Fraction getZero() {
        return Fraction.ZERO;
    }

    
    public Class<? extends FieldElement<Fraction>> getRuntimeClass() {
        return Fraction.class;
    }
    // CHECKSTYLE: stop HideUtilityClassConstructor
    
    private static class LazyHolder {
        
        private static final FractionField INSTANCE = new FractionField();
    }
    // CHECKSTYLE: resume HideUtilityClassConstructor

    
    private Object readResolve() {
        // return the singleton instance
        return LazyHolder.INSTANCE;
    }

}
