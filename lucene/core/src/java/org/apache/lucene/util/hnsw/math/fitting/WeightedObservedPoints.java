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
package org.apache.lucene.util.hnsw.math.fitting;

import java.util.List;
import java.util.ArrayList;
import java.io.Serializable;


public class WeightedObservedPoints implements Serializable {
    
    private static final long serialVersionUID = 20130813L;

    
    private final List<WeightedObservedPoint> observations
        = new ArrayList<WeightedObservedPoint>();

    
    public void add(double x, double y) {
        add(1d, x, y);
    }

    
    public void add(double weight, double x, double y) {
        observations.add(new WeightedObservedPoint(weight, x, y));
    }

    
    public void add(WeightedObservedPoint observed) {
        observations.add(observed);
    }

    
    public List<WeightedObservedPoint> toList() {
        // The copy is necessary to ensure thread-safety because of the
        // "clear" method (which otherwise would be able to empty the
        // list of points while it is being used by another thread).
        return new ArrayList<WeightedObservedPoint>(observations);
    }

    
    public void clear() {
        observations.clear();
    }
}
