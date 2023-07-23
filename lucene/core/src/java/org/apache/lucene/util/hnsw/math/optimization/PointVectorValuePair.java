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

package org.apache.lucene.util.hnsw.math.optimization;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.util.Pair;


@Deprecated
public class PointVectorValuePair extends Pair<double[], double[]> implements Serializable {

    
    private static final long serialVersionUID = 20120513L;

    
    public PointVectorValuePair(final double[] point,
                                final double[] value) {
        this(point, value, true);
    }

    
    public PointVectorValuePair(final double[] point,
                                final double[] value,
                                final boolean copyArray) {
        super(copyArray ?
              ((point == null) ? null :
               point.clone()) :
              point,
              copyArray ?
              ((value == null) ? null :
               value.clone()) :
              value);
    }

    
    public double[] getPoint() {
        final double[] p = getKey();
        return p == null ? null : p.clone();
    }

    
    public double[] getPointRef() {
        return getKey();
    }

    
    @Override
    public double[] getValue() {
        final double[] v = super.getValue();
        return v == null ? null : v.clone();
    }

    
    public double[] getValueRef() {
        return super.getValue();
    }

    
    private Object writeReplace() {
        return new DataTransferObject(getKey(), getValue());
    }

    
    private static class DataTransferObject implements Serializable {
        
        private static final long serialVersionUID = 20120513L;
        
        private final double[] point;
        
        private final double[] value;

        
        DataTransferObject(final double[] point, final double[] value) {
            this.point = point.clone();
            this.value = value.clone();
        }

        
        private Object readResolve() {
            return new PointVectorValuePair(point, value, false);
        }
    }
}
