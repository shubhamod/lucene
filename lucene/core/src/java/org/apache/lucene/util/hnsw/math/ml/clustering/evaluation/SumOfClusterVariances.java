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

package org.apache.lucene.util.hnsw.math.ml.clustering.evaluation;

import java.util.List;

import org.apache.lucene.util.hnsw.math.ml.clustering.Cluster;
import org.apache.lucene.util.hnsw.math.ml.clustering.Clusterable;
import org.apache.lucene.util.hnsw.math.ml.distance.DistanceMeasure;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.Variance;


public class SumOfClusterVariances<T extends Clusterable> extends ClusterEvaluator<T> {

    
    public SumOfClusterVariances(final DistanceMeasure measure) {
        super(measure);
    }

    
    @Override
    public double score(final List<? extends Cluster<T>> clusters) {
        double varianceSum = 0.0;
        for (final Cluster<T> cluster : clusters) {
            if (!cluster.getPoints().isEmpty()) {

                final Clusterable center = centroidOf(cluster);

                // compute the distance variance of the current cluster
                final Variance stat = new Variance();
                for (final T point : cluster.getPoints()) {
                    stat.increment(distance(point, center));
                }
                varianceSum += stat.getResult();

            }
        }
        return varianceSum;
    }

}
