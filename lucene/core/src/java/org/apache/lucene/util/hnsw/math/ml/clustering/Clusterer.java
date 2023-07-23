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

import java.util.Collection;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.ml.distance.DistanceMeasure;


public abstract class Clusterer<T extends Clusterable> {

    
    private DistanceMeasure measure;

    
    protected Clusterer(final DistanceMeasure measure) {
        this.measure = measure;
    }

    
    public abstract List<? extends Cluster<T>> cluster(Collection<T> points)
            throws MathIllegalArgumentException, ConvergenceException;

    
    public DistanceMeasure getDistanceMeasure() {
        return measure;
    }

    
    protected double distance(final Clusterable p1, final Clusterable p2) {
        return measure.compute(p1.getPoint(), p2.getPoint());
    }

}
