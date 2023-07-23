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
package org.apache.lucene.util.hnsw.math.distribution;

import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.util.Pair;


public class MixtureMultivariateNormalDistribution
    extends MixtureMultivariateRealDistribution<MultivariateNormalDistribution> {

    
    public MixtureMultivariateNormalDistribution(double[] weights,
                                                 double[][] means,
                                                 double[][][] covariances) {
        super(createComponents(weights, means, covariances));
    }

    
    public MixtureMultivariateNormalDistribution(List<Pair<Double, MultivariateNormalDistribution>> components) {
        super(components);
    }

    
    public MixtureMultivariateNormalDistribution(RandomGenerator rng,
                                                 List<Pair<Double, MultivariateNormalDistribution>> components)
        throws NotPositiveException, DimensionMismatchException {
        super(rng, components);
    }

    
    private static List<Pair<Double, MultivariateNormalDistribution>> createComponents(double[] weights,
                                                                                       double[][] means,
                                                                                       double[][][] covariances) {
        final List<Pair<Double, MultivariateNormalDistribution>> mvns
            = new ArrayList<Pair<Double, MultivariateNormalDistribution>>(weights.length);

        for (int i = 0; i < weights.length; i++) {
            final MultivariateNormalDistribution dist
                = new MultivariateNormalDistribution(means[i], covariances[i]);

            mvns.add(new Pair<Double, MultivariateNormalDistribution>(weights[i], dist));
        }

        return mvns;
    }
}
