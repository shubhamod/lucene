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

package org.apache.lucene.util.hnsw.math.optimization.fitting;

import java.io.Serializable;


@Deprecated
public class WeightedObservedPoint implements Serializable {

    
    private static final long serialVersionUID = 5306874947404636157L;

    
    private final double weight;

    
    private final double x;

    
    private final double y;

    
    public WeightedObservedPoint(final double weight, final double x, final double y) {
        this.weight = weight;
        this.x      = x;
        this.y      = y;
    }

    
    public double getWeight() {
        return weight;
    }

    
    public double getX() {
        return x;
    }

    
    public double getY() {
        return y;
    }

}

