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

package org.apache.lucene.util.hnsw.math.ml.clustering;

import java.io.Serializable;
import java.util.Arrays;


public class DoublePoint implements Clusterable, Serializable {

    
    private static final long serialVersionUID = 3946024775784901369L;

    
    private final double[] point;

    
    public DoublePoint(final double[] point) {
        this.point = point;
    }

    
    public DoublePoint(final int[] point) {
        this.point = new double[point.length];
        for ( int i = 0; i < point.length; i++) {
            this.point[i] = point[i];
        }
    }

    
    public double[] getPoint() {
        return point;
    }

    
    @Override
    public boolean equals(final Object other) {
        if (!(other instanceof DoublePoint)) {
            return false;
        }
        return Arrays.equals(point, ((DoublePoint) other).point);
    }

    
    @Override
    public int hashCode() {
        return Arrays.hashCode(point);
    }

    
    @Override
    public String toString() {
        return Arrays.toString(point);
    }

}
