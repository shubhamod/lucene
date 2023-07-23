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

import java.util.Map;
import java.util.HashMap;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.Neuron;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.twod.NeuronSquareMesh2D;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;


public class LocationFinder {
    
    private final Map<Long, Location> locations = new HashMap<Long, Location>();

    
    public static class Location {
        
        private final int row;
        
        private final int column;

        
        public Location(int row,
                        int column) {
            this.row = row;
            this.column = column;
        }

        
        public int getRow() {
            return row;
        }

        
        public int getColumn() {
            return column;
        }
    }

    
    public LocationFinder(NeuronSquareMesh2D map) {
        final int nR = map.getNumberOfRows();
        final int nC = map.getNumberOfColumns();

        for (int r = 0; r < nR; r++) {
            for (int c = 0; c < nC; c++) {
                final Long id = map.getNeuron(r, c).getIdentifier();
                if (locations.get(id) != null) {
                    throw new MathIllegalStateException();
                }
                locations.put(id, new Location(r, c));
            }
        }
    }

    
    public Location getLocation(Neuron n) {
        return locations.get(n.getIdentifier());
    }
}
