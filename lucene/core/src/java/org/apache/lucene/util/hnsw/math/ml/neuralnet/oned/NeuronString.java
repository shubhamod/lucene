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

package org.apache.lucene.util.hnsw.math.ml.neuralnet.oned;

import java.io.Serializable;
import java.io.ObjectInputStream;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.Network;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.FeatureInitializer;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;


public class NeuronString implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final Network network;
    
    private final int size;
    
    private final boolean wrap;

    
    private final long[] identifiers;

    
    NeuronString(boolean wrap,
                 double[][] featuresList) {
        size = featuresList.length;

        if (size < 2) {
            throw new NumberIsTooSmallException(size, 2, true);
        }

        this.wrap = wrap;

        final int fLen = featuresList[0].length;
        network = new Network(0, fLen);
        identifiers = new long[size];

        // Add neurons.
        for (int i = 0; i < size; i++) {
            identifiers[i] = network.createNeuron(featuresList[i]);
        }

        // Add links.
        createLinks();
    }

    
    public NeuronString(int num,
                        boolean wrap,
                        FeatureInitializer[] featureInit) {
        if (num < 2) {
            throw new NumberIsTooSmallException(num, 2, true);
        }

        size = num;
        this.wrap = wrap;
        identifiers = new long[num];

        final int fLen = featureInit.length;
        network = new Network(0, fLen);

        // Add neurons.
        for (int i = 0; i < num; i++) {
            final double[] features = new double[fLen];
            for (int fIndex = 0; fIndex < fLen; fIndex++) {
                features[fIndex] = featureInit[fIndex].value();
            }
            identifiers[i] = network.createNeuron(features);
        }

        // Add links.
        createLinks();
    }

    
    public Network getNetwork() {
        return network;
    }

    
    public int getSize() {
        return size;
    }

    
    public double[] getFeatures(int i) {
        if (i < 0 ||
            i >= size) {
            throw new OutOfRangeException(i, 0, size - 1);
        }

        return network.getNeuron(identifiers[i]).getFeatures();
    }

    
    private void createLinks() {
        for (int i = 0; i < size - 1; i++) {
            network.addLink(network.getNeuron(i), network.getNeuron(i + 1));
        }
        for (int i = size - 1; i > 0; i--) {
            network.addLink(network.getNeuron(i), network.getNeuron(i - 1));
        }
        if (wrap) {
            network.addLink(network.getNeuron(0), network.getNeuron(size - 1));
            network.addLink(network.getNeuron(size - 1), network.getNeuron(0));
        }
    }

    
    private void readObject(ObjectInputStream in) {
        throw new IllegalStateException();
    }

    
    private Object writeReplace() {
        final double[][] featuresList = new double[size][];
        for (int i = 0; i < size; i++) {
            featuresList[i] = getFeatures(i);
        }

        return new SerializationProxy(wrap,
                                      featuresList);
    }

    
    private static class SerializationProxy implements Serializable {
        
        private static final long serialVersionUID = 20130226L;
        
        private final boolean wrap;
        
        private final double[][] featuresList;

        
        SerializationProxy(boolean wrap,
                           double[][] featuresList) {
            this.wrap = wrap;
            this.featuresList = featuresList;
        }

        
        private Object readResolve() {
            return new NeuronString(wrap,
                                    featuresList);
        }
    }
}
