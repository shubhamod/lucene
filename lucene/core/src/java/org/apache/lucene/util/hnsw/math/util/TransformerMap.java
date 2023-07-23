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
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;


public class TransformerMap implements NumberTransformer, Serializable {

    
    private static final long serialVersionUID = 4605318041528645258L;

    
    private NumberTransformer defaultTransformer = null;

    
    private Map<Class<?>, NumberTransformer> map = null;

    
    public TransformerMap() {
        map = new HashMap<Class<?>, NumberTransformer>();
        defaultTransformer = new DefaultTransformer();
    }

    
    public boolean containsClass(Class<?> key) {
        return map.containsKey(key);
    }

    
    public boolean containsTransformer(NumberTransformer value) {
        return map.containsValue(value);
    }

    
    public NumberTransformer getTransformer(Class<?> key) {
        return map.get(key);
    }

    
    public NumberTransformer putTransformer(Class<?> key, NumberTransformer transformer) {
        return map.put(key, transformer);
    }

    
    public NumberTransformer removeTransformer(Class<?> key) {
        return map.remove(key);
    }

    
    public void clear() {
        map.clear();
    }

    
    public Set<Class<?>> classes() {
        return map.keySet();
    }

    
    public Collection<NumberTransformer> transformers() {
        return map.values();
    }

    
    public double transform(Object o) throws MathIllegalArgumentException {
        double value = Double.NaN;

        if (o instanceof Number || o instanceof String) {
            value = defaultTransformer.transform(o);
        } else {
            NumberTransformer trans = getTransformer(o.getClass());
            if (trans != null) {
                value = trans.transform(o);
            }
        }

        return value;
    }

    
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other instanceof TransformerMap) {
            TransformerMap rhs = (TransformerMap) other;
            if (! defaultTransformer.equals(rhs.defaultTransformer)) {
                return false;
            }
            if (map.size() != rhs.map.size()) {
                return false;
            }
            for (Map.Entry<Class<?>, NumberTransformer> entry : map.entrySet()) {
                if (! entry.getValue().equals(rhs.map.get(entry.getKey()))) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    
    @Override
    public int hashCode() {
        int hash = defaultTransformer.hashCode();
        for (NumberTransformer t : map.values()) {
            hash = hash * 31 + t.hashCode();
        }
        return hash;
    }

}
