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
import java.util.ArrayList;
import java.util.List;


public class Cluster<T extends Clusterable> implements Serializable {

    
    private static final long serialVersionUID = -3442297081515880464L;

    
    private final List<T> points;

    
    public Cluster() {
        points = new ArrayList<T>();
    }

    
    public void addPoint(final T point) {
        points.add(point);
    }

    
    public List<T> getPoints() {
        return points;
    }

}
