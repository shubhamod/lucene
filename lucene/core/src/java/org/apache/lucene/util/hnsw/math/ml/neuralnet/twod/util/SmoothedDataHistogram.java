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

package org.apache.lucene.util.hnsw.math.ml.neuralnet.twod.util;

import org.apache.lucene.util.hnsw.math.ml.neuralnet.MapUtils;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.Neuron;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.twod.NeuronSquareMesh2D;
import org.apache.lucene.util.hnsw.math.ml.distance.DistanceMeasure;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;


public class SmoothedDataHistogram implements MapDataVisualization {
    
    private final int smoothingBins;
    
    private final DistanceMeasure distance;
    
    private final double membershipNormalization;

    
    public SmoothedDataHistogram(int smoothingBins,
                                 DistanceMeasure distance) {
        this.smoothingBins = smoothingBins;
        this.distance = distance;

        double sum = 0;
        for (int i = 0; i < smoothingBins; i++) {
            sum += smoothingBins - i;
        }

        this.membershipNormalization = 1d / sum;
    }

    
    public double[][] computeImage(NeuronSquareMesh2D map,
                                   Iterable<double[]> data) {
        final int nR = map.getNumberOfRows();
        final int nC = map.getNumberOfColumns();

        final int mapSize = nR * nC;
        if (mapSize < smoothingBins) {
            throw new NumberIsTooSmallException(mapSize, smoothingBins, true);
        }

        final LocationFinder finder = new LocationFinder(map);

        // Histogram bins.
        final double[][] histo = new double[nR][nC];

        for (double[] sample : data) {
            final Neuron[] sorted = MapUtils.sort(sample,
                                                  map.getNetwork(),
                                                  distance);
            for (int i = 0; i < smoothingBins; i++) {
                final LocationFinder.Location loc = finder.getLocation(sorted[i]);
                final int row = loc.getRow();
                final int col = loc.getColumn();
                histo[row][col] += (smoothingBins - i) * membershipNormalization;
            }
        }

        return histo;
    }
}
