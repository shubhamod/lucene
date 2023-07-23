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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;


public class Decimal64Field implements Field<Decimal64> {

    
    private static final Decimal64Field INSTANCE = new Decimal64Field();

    
    private Decimal64Field() {
        // Do nothing
    }

    
    public static final Decimal64Field getInstance() {
        return INSTANCE;
    }

    
    public Decimal64 getZero() {
        return Decimal64.ZERO;
    }

    
    public Decimal64 getOne() {
        return Decimal64.ONE;
    }

    
    public Class<? extends FieldElement<Decimal64>> getRuntimeClass() {
        return Decimal64.class;
    }
}
