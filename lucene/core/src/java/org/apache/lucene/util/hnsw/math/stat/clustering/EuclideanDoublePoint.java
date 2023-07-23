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
package org.apache.lucene.util.hnsw.math.stat.clustering;

import java.io.Serializable;
import java.util.Collection;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.util.MathArrays;


@Deprecated
public class EuclideanDoublePoint implements Clusterable<EuclideanDoublePoint>, Serializable {

    
    private static final long serialVersionUID = 8026472786091227632L;

    
    private final double[] point;

    
    public EuclideanDoublePoint(final double[] point) {
        this.point = point;
    }

    
    public EuclideanDoublePoint centroidOf(final Collection<EuclideanDoublePoint> points) {
        final double[] centroid = new double[getPoint().length];
        for (final EuclideanDoublePoint p : points) {
            for (int i = 0; i < centroid.length; i++) {
                centroid[i] += p.getPoint()[i];
            }
        }
        for (int i = 0; i < centroid.length; i++) {
            centroid[i] /= points.size();
        }
        return new EuclideanDoublePoint(centroid);
    }

    
    public double distanceFrom(final EuclideanDoublePoint p) {
        return MathArrays.distance(point, p.getPoint());
    }

    
    @Override
    public boolean equals(final Object other) {
        if (!(other instanceof EuclideanDoublePoint)) {
            return false;
        }
        return Arrays.equals(point, ((EuclideanDoublePoint) other).point);
    }

    
    public double[] getPoint() {
        return point;
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
