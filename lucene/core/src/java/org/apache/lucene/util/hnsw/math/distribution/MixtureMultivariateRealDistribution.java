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
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.Pair;


public class MixtureMultivariateRealDistribution<T extends MultivariateRealDistribution>
    extends AbstractMultivariateRealDistribution {
    
    private final double[] weight;
    
    private final List<T> distribution;

    
    public MixtureMultivariateRealDistribution(List<Pair<Double, T>> components) {
        this(new Well19937c(), components);
    }

    
    public MixtureMultivariateRealDistribution(RandomGenerator rng,
                                               List<Pair<Double, T>> components) {
        super(rng, components.get(0).getSecond().getDimension());

        final int numComp = components.size();
        final int dim = getDimension();
        double weightSum = 0;
        for (int i = 0; i < numComp; i++) {
            final Pair<Double, T> comp = components.get(i);
            if (comp.getSecond().getDimension() != dim) {
                throw new DimensionMismatchException(comp.getSecond().getDimension(), dim);
            }
            if (comp.getFirst() < 0) {
                throw new NotPositiveException(comp.getFirst());
            }
            weightSum += comp.getFirst();
        }

        // Check for overflow.
        if (Double.isInfinite(weightSum)) {
            throw new MathArithmeticException(LocalizedFormats.OVERFLOW);
        }

        // Store each distribution and its normalized weight.
        distribution = new ArrayList<T>();
        weight = new double[numComp];
        for (int i = 0; i < numComp; i++) {
            final Pair<Double, T> comp = components.get(i);
            weight[i] = comp.getFirst() / weightSum;
            distribution.add(comp.getSecond());
        }
    }

    
    public double density(final double[] values) {
        double p = 0;
        for (int i = 0; i < weight.length; i++) {
            p += weight[i] * distribution.get(i).density(values);
        }
        return p;
    }

    
    @Override
    public double[] sample() {
        // Sampled values.
        double[] vals = null;

        // Determine which component to sample from.
        final double randomValue = random.nextDouble();
        double sum = 0;

        for (int i = 0; i < weight.length; i++) {
            sum += weight[i];
            if (randomValue <= sum) {
                // pick model i
                vals = distribution.get(i).sample();
                break;
            }
        }

        if (vals == null) {
            // This should never happen, but it ensures we won't return a null in
            // case the loop above has some floating point inequality problem on
            // the final iteration.
            vals = distribution.get(weight.length - 1).sample();
        }

        return vals;
    }

    
    @Override
    public void reseedRandomGenerator(long seed) {
        // Seed needs to be propagated to underlying components
        // in order to maintain consistency between runs.
        super.reseedRandomGenerator(seed);

        for (int i = 0; i < distribution.size(); i++) {
            // Make each component's seed different in order to avoid
            // using the same sequence of random numbers.
            distribution.get(i).reseedRandomGenerator(i + 1 + seed);
        }
    }

    
    public List<Pair<Double, T>> getComponents() {
        final List<Pair<Double, T>> list = new ArrayList<Pair<Double, T>>(weight.length);

        for (int i = 0; i < weight.length; i++) {
            list.add(new Pair<Double, T>(weight[i], distribution.get(i)));
        }

        return list;
    }
}
