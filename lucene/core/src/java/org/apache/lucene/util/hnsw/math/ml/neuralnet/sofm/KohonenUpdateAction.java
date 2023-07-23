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

package org.apache.lucene.util.hnsw.math.ml.neuralnet.sofm;

import java.util.Collection;
import java.util.HashSet;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.lucene.util.hnsw.math.analysis.function.Gaussian;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.ml.distance.DistanceMeasure;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.MapUtils;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.Network;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.Neuron;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.UpdateAction;


public class KohonenUpdateAction implements UpdateAction {
    
    private final DistanceMeasure distance;
    
    private final LearningFactorFunction learningFactor;
    
    private final NeighbourhoodSizeFunction neighbourhoodSize;
    
    private final AtomicLong numberOfCalls = new AtomicLong(0);

    
    public KohonenUpdateAction(DistanceMeasure distance,
                               LearningFactorFunction learningFactor,
                               NeighbourhoodSizeFunction neighbourhoodSize) {
        this.distance = distance;
        this.learningFactor = learningFactor;
        this.neighbourhoodSize = neighbourhoodSize;
    }

    
    public void update(Network net,
                       double[] features) {
        final long numCalls = numberOfCalls.incrementAndGet() - 1;
        final double currentLearning = learningFactor.value(numCalls);
        final Neuron best = findAndUpdateBestNeuron(net,
                                                    features,
                                                    currentLearning);

        final int currentNeighbourhood = neighbourhoodSize.value(numCalls);
        // The farther away the neighbour is from the winning neuron, the
        // smaller the learning rate will become.
        final Gaussian neighbourhoodDecay
            = new Gaussian(currentLearning,
                           0,
                           currentNeighbourhood);

        if (currentNeighbourhood > 0) {
            // Initial set of neurons only contains the winning neuron.
            Collection<Neuron> neighbours = new HashSet<Neuron>();
            neighbours.add(best);
            // Winning neuron must be excluded from the neighbours.
            final HashSet<Neuron> exclude = new HashSet<Neuron>();
            exclude.add(best);

            int radius = 1;
            do {
                // Retrieve immediate neighbours of the current set of neurons.
                neighbours = net.getNeighbours(neighbours, exclude);

                // Update all the neighbours.
                for (Neuron n : neighbours) {
                    updateNeighbouringNeuron(n, features, neighbourhoodDecay.value(radius));
                }

                // Add the neighbours to the exclude list so that they will
                // not be update more than once per training step.
                exclude.addAll(neighbours);
                ++radius;
            } while (radius <= currentNeighbourhood);
        }
    }

    
    public long getNumberOfCalls() {
        return numberOfCalls.get();
    }

    
    private boolean attemptNeuronUpdate(Neuron n,
                                        double[] features,
                                        double learningRate) {
        final double[] expect = n.getFeatures();
        final double[] update = computeFeatures(expect,
                                                features,
                                                learningRate);

        return n.compareAndSetFeatures(expect, update);
    }

    
    private void updateNeighbouringNeuron(Neuron n,
                                          double[] features,
                                          double learningRate) {
        while (true) {
            if (attemptNeuronUpdate(n, features, learningRate)) {
                break;
            }
        }
    }

    
    private Neuron findAndUpdateBestNeuron(Network net,
                                           double[] features,
                                           double learningRate) {
        while (true) {
            final Neuron best = MapUtils.findBest(features, net, distance);

            if (attemptNeuronUpdate(best, features, learningRate)) {
                return best;
            }

            // If another thread modified the state of the winning neuron,
            // it may not be the best match anymore for the given training
            // sample: Hence, the winner search is performed again.
        }
    }

    
    private double[] computeFeatures(double[] current,
                                     double[] sample,
                                     double learningRate) {
        final ArrayRealVector c = new ArrayRealVector(current, false);
        final ArrayRealVector s = new ArrayRealVector(sample, false);
        // c + learningRate * (s - c)
        return s.subtract(c).mapMultiplyToSelf(learningRate).add(c).toArray();
    }
}
