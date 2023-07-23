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

package org.apache.lucene.util.hnsw.math.ml.neuralnet;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Comparator;

import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.ml.distance.DistanceMeasure;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.twod.NeuronSquareMesh2D;
import org.apache.lucene.util.hnsw.math.util.Pair;


public class MapUtils {
    
    private MapUtils() {}

    
    public static Neuron findBest(double[] features,
                                  Iterable<Neuron> neurons,
                                  DistanceMeasure distance) {
        Neuron best = null;
        double min = Double.POSITIVE_INFINITY;
        for (final Neuron n : neurons) {
            final double d = distance.compute(n.getFeatures(), features);
            if (d < min) {
                min = d;
                best = n;
            }
        }

        return best;
    }

    
    public static Pair<Neuron, Neuron> findBestAndSecondBest(double[] features,
                                                             Iterable<Neuron> neurons,
                                                             DistanceMeasure distance) {
        Neuron[] best = { null, null };
        double[] min = { Double.POSITIVE_INFINITY,
                         Double.POSITIVE_INFINITY };
        for (final Neuron n : neurons) {
            final double d = distance.compute(n.getFeatures(), features);
            if (d < min[0]) {
                // Replace second best with old best.
                min[1] = min[0];
                best[1] = best[0];

                // Store current as new best.
                min[0] = d;
                best[0] = n;
            } else if (d < min[1]) {
                // Replace old second best with current.
                min[1] = d;
                best[1] = n;
            }
        }

        return new Pair<Neuron, Neuron>(best[0], best[1]);
    }

    
    public static Neuron[] sort(double[] features,
                                Iterable<Neuron> neurons,
                                DistanceMeasure distance) {
        final List<PairNeuronDouble> list = new ArrayList<PairNeuronDouble>();

        for (final Neuron n : neurons) {
            final double d = distance.compute(n.getFeatures(), features);
            list.add(new PairNeuronDouble(n, d));
        }

        Collections.sort(list, PairNeuronDouble.COMPARATOR);

        final int len = list.size();
        final Neuron[] sorted = new Neuron[len];

        for (int i = 0; i < len; i++) {
            sorted[i] = list.get(i).getNeuron();
        }
        return sorted;
    }

    
    public static double[][] computeU(NeuronSquareMesh2D map,
                                      DistanceMeasure distance) {
        final int numRows = map.getNumberOfRows();
        final int numCols = map.getNumberOfColumns();
        final double[][] uMatrix = new double[numRows][numCols];

        final Network net = map.getNetwork();

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                final Neuron neuron = map.getNeuron(i, j);
                final Collection<Neuron> neighbours = net.getNeighbours(neuron);
                final double[] features = neuron.getFeatures();

                double d = 0;
                int count = 0;
                for (Neuron n : neighbours) {
                    ++count;
                    d += distance.compute(features, n.getFeatures());
                }

                uMatrix[i][j] = d / count;
            }
        }

        return uMatrix;
    }

    
    public static int[][] computeHitHistogram(Iterable<double[]> data,
                                              NeuronSquareMesh2D map,
                                              DistanceMeasure distance) {
        final HashMap<Neuron, Integer> hit = new HashMap<Neuron, Integer>();
        final Network net = map.getNetwork();

        for (double[] f : data) {
            final Neuron best = findBest(f, net, distance);
            final Integer count = hit.get(best);
            if (count == null) {
                hit.put(best, 1);
            } else {
                hit.put(best, count + 1);
            }
        }

        // Copy the histogram data into a 2D map.
        final int numRows = map.getNumberOfRows();
        final int numCols = map.getNumberOfColumns();
        final int[][] histo = new int[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                final Neuron neuron = map.getNeuron(i, j);
                final Integer count = hit.get(neuron);
                if (count == null) {
                    histo[i][j] = 0;
                } else {
                    histo[i][j] = count;
                }
            }
        }

        return histo;
    }

    
    public static double computeQuantizationError(Iterable<double[]> data,
                                                  Iterable<Neuron> neurons,
                                                  DistanceMeasure distance) {
        double d = 0;
        int count = 0;
        for (double[] f : data) {
            ++count;
            d += distance.compute(f, findBest(f, neurons, distance).getFeatures());
        }

        if (count == 0) {
            throw new NoDataException();
        }

        return d / count;
    }

    
    public static double computeTopographicError(Iterable<double[]> data,
                                                 Network net,
                                                 DistanceMeasure distance) {
        int notAdjacentCount = 0;
        int count = 0;
        for (double[] f : data) {
            ++count;
            final Pair<Neuron, Neuron> p = findBestAndSecondBest(f, net, distance);
            if (!net.getNeighbours(p.getFirst()).contains(p.getSecond())) {
                // Increment count if first and second best matching units
                // are not neighbours.
                ++notAdjacentCount;
            }
        }

        if (count == 0) {
            throw new NoDataException();
        }

        return ((double) notAdjacentCount) / count;
    }

    
    private static class PairNeuronDouble {
        
        static final Comparator<PairNeuronDouble> COMPARATOR
            = new Comparator<PairNeuronDouble>() {
            
            public int compare(PairNeuronDouble o1,
                               PairNeuronDouble o2) {
                return Double.compare(o1.value, o2.value);
            }
        };
        
        private final Neuron neuron;
        
        private final double value;

        
        PairNeuronDouble(Neuron neuron, double value) {
            this.neuron = neuron;
            this.value = value;
        }

        
        public Neuron getNeuron() {
            return neuron;
        }

    }
}
