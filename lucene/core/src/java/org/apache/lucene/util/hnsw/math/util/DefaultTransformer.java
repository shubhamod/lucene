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

import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;


public class DefaultTransformer implements NumberTransformer, Serializable {

    
    private static final long serialVersionUID = 4019938025047800455L;

    
    public double transform(Object o)
        throws NullArgumentException, MathIllegalArgumentException {

        if (o == null) {
            throw new NullArgumentException(LocalizedFormats.OBJECT_TRANSFORMATION);
        }

        if (o instanceof Number) {
            return ((Number)o).doubleValue();
        }

        try {
            return Double.parseDouble(o.toString());
        } catch (NumberFormatException e) {
            throw new MathIllegalArgumentException(LocalizedFormats.CANNOT_TRANSFORM_TO_DOUBLE,
                                                   o.toString());
        }
    }

    
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        return other instanceof DefaultTransformer;
    }

    
    @Override
    public int hashCode() {
        // some arbitrary number ...
        return 401993047;
    }

}
