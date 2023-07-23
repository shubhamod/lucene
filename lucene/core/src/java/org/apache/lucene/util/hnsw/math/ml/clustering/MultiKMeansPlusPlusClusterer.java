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
import org.apache.lucene.util.hnsw.math.ml.clustering.evaluation.ClusterEvaluator;
import org.apache.lucene.util.hnsw.math.ml.clustering.evaluation.SumOfClusterVariances;


public class MultiKMeansPlusPlusClusterer<T extends Clusterable> extends Clusterer<T> {

    
    private final KMeansPlusPlusClusterer<T> clusterer;

    
    private final int numTrials;

    
    private final ClusterEvaluator<T> evaluator;

    
    public MultiKMeansPlusPlusClusterer(final KMeansPlusPlusClusterer<T> clusterer,
                                        final int numTrials) {
        this(clusterer, numTrials, new SumOfClusterVariances<T>(clusterer.getDistanceMeasure()));
    }

    
    public MultiKMeansPlusPlusClusterer(final KMeansPlusPlusClusterer<T> clusterer,
                                        final int numTrials,
                                        final ClusterEvaluator<T> evaluator) {
        super(clusterer.getDistanceMeasure());
        this.clusterer = clusterer;
        this.numTrials = numTrials;
        this.evaluator = evaluator;
    }

    
    public KMeansPlusPlusClusterer<T> getClusterer() {
        return clusterer;
    }

    
    public int getNumTrials() {
        return numTrials;
    }

    
    public ClusterEvaluator<T> getClusterEvaluator() {
       return evaluator;
    }

    
    @Override
    public List<CentroidCluster<T>> cluster(final Collection<T> points)
        throws MathIllegalArgumentException, ConvergenceException {

        // at first, we have not found any clusters list yet
        List<CentroidCluster<T>> best = null;
        double bestVarianceSum = Double.POSITIVE_INFINITY;

        // do several clustering trials
        for (int i = 0; i < numTrials; ++i) {

            // compute a clusters list
            List<CentroidCluster<T>> clusters = clusterer.cluster(points);

            // compute the variance of the current list
            final double varianceSum = evaluator.score(clusters);

            if (evaluator.isBetterScore(varianceSum, bestVarianceSum)) {
                // this one is the best we have found so far, remember it
                best            = clusters;
                bestVarianceSum = varianceSum;
            }

        }

        // return the best clusters list found
        return best;

    }

}
