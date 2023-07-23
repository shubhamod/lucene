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

import java.io.Serializable;
import java.io.ObjectInputStream;
import java.util.NoSuchElementException;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
import java.util.Collection;
import java.util.Iterator;
import java.util.Comparator;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;


public class Network
    implements Iterable<Neuron>,
               Serializable {
    
    private static final long serialVersionUID = 20130207L;
    
    private final ConcurrentHashMap<Long, Neuron> neuronMap
        = new ConcurrentHashMap<Long, Neuron>();
    
    private final AtomicLong nextId;
    
    private final int featureSize;
    
    private final ConcurrentHashMap<Long, Set<Long>> linkMap
        = new ConcurrentHashMap<Long, Set<Long>>();

    
    public static class NeuronIdentifierComparator
        implements Comparator<Neuron>,
                   Serializable {
        
        private static final long serialVersionUID = 20130207L;

        
        public int compare(Neuron a,
                           Neuron b) {
            final long aId = a.getIdentifier();
            final long bId = b.getIdentifier();
            return aId < bId ? -1 :
                aId > bId ? 1 : 0;
        }
    }

    
    Network(long nextId,
            int featureSize,
            Neuron[] neuronList,
            long[][] neighbourIdList) {
        final int numNeurons = neuronList.length;
        if (numNeurons != neighbourIdList.length) {
            throw new MathIllegalStateException();
        }

        for (int i = 0; i < numNeurons; i++) {
            final Neuron n = neuronList[i];
            final long id = n.getIdentifier();
            if (id >= nextId) {
                throw new MathIllegalStateException();
            }
            neuronMap.put(id, n);
            linkMap.put(id, new HashSet<Long>());
        }

        for (int i = 0; i < numNeurons; i++) {
            final long aId = neuronList[i].getIdentifier();
            final Set<Long> aLinks = linkMap.get(aId);
            for (Long bId : neighbourIdList[i]) {
                if (neuronMap.get(bId) == null) {
                    throw new MathIllegalStateException();
                }
                addLinkToLinkSet(aLinks, bId);
            }
        }

        this.nextId = new AtomicLong(nextId);
        this.featureSize = featureSize;
    }

    
    public Network(long initialIdentifier,
                   int featureSize) {
        nextId = new AtomicLong(initialIdentifier);
        this.featureSize = featureSize;
    }

    
    public synchronized Network copy() {
        final Network copy = new Network(nextId.get(),
                                         featureSize);


        for (Map.Entry<Long, Neuron> e : neuronMap.entrySet()) {
            copy.neuronMap.put(e.getKey(), e.getValue().copy());
        }

        for (Map.Entry<Long, Set<Long>> e : linkMap.entrySet()) {
            copy.linkMap.put(e.getKey(), new HashSet<Long>(e.getValue()));
        }

        return copy;
    }

    
    public Iterator<Neuron> iterator() {
        return neuronMap.values().iterator();
    }

    
    public Collection<Neuron> getNeurons(Comparator<Neuron> comparator) {
        final List<Neuron> neurons = new ArrayList<Neuron>();
        neurons.addAll(neuronMap.values());

        Collections.sort(neurons, comparator);

        return neurons;
    }

    
    public long createNeuron(double[] features) {
        if (features.length != featureSize) {
            throw new DimensionMismatchException(features.length, featureSize);
        }

        final long id = createNextId();
        neuronMap.put(id, new Neuron(id, features));
        linkMap.put(id, new HashSet<Long>());
        return id;
    }

    
    public void deleteNeuron(Neuron neuron) {
        final Collection<Neuron> neighbours = getNeighbours(neuron);

        // Delete links to from neighbours.
        for (Neuron n : neighbours) {
            deleteLink(n, neuron);
        }

        // Remove neuron.
        neuronMap.remove(neuron.getIdentifier());
    }

    
    public int getFeaturesSize() {
        return featureSize;
    }

    
    public void addLink(Neuron a,
                        Neuron b) {
        final long aId = a.getIdentifier();
        final long bId = b.getIdentifier();

        // Check that the neurons belong to this network.
        if (a != getNeuron(aId)) {
            throw new NoSuchElementException(Long.toString(aId));
        }
        if (b != getNeuron(bId)) {
            throw new NoSuchElementException(Long.toString(bId));
        }

        // Add link from "a" to "b".
        addLinkToLinkSet(linkMap.get(aId), bId);
    }

    
    private void addLinkToLinkSet(Set<Long> linkSet,
                                  long id) {
        linkSet.add(id);
    }

    
    public void deleteLink(Neuron a,
                           Neuron b) {
        final long aId = a.getIdentifier();
        final long bId = b.getIdentifier();

        // Check that the neurons belong to this network.
        if (a != getNeuron(aId)) {
            throw new NoSuchElementException(Long.toString(aId));
        }
        if (b != getNeuron(bId)) {
            throw new NoSuchElementException(Long.toString(bId));
        }

        // Delete link from "a" to "b".
        deleteLinkFromLinkSet(linkMap.get(aId), bId);
    }

    
    private void deleteLinkFromLinkSet(Set<Long> linkSet,
                                       long id) {
        linkSet.remove(id);
    }

    
    public Neuron getNeuron(long id) {
        final Neuron n = neuronMap.get(id);
        if (n == null) {
            throw new NoSuchElementException(Long.toString(id));
        }
        return n;
    }

    
    public Collection<Neuron> getNeighbours(Iterable<Neuron> neurons) {
        return getNeighbours(neurons, null);
    }

    
    public Collection<Neuron> getNeighbours(Iterable<Neuron> neurons,
                                            Iterable<Neuron> exclude) {
        final Set<Long> idList = new HashSet<Long>();

        for (Neuron n : neurons) {
            idList.addAll(linkMap.get(n.getIdentifier()));
        }
        if (exclude != null) {
            for (Neuron n : exclude) {
                idList.remove(n.getIdentifier());
            }
        }

        final List<Neuron> neuronList = new ArrayList<Neuron>();
        for (Long id : idList) {
            neuronList.add(getNeuron(id));
        }

        return neuronList;
    }

    
    public Collection<Neuron> getNeighbours(Neuron neuron) {
        return getNeighbours(neuron, null);
    }

    
    public Collection<Neuron> getNeighbours(Neuron neuron,
                                            Iterable<Neuron> exclude) {
        final Set<Long> idList = linkMap.get(neuron.getIdentifier());
        if (exclude != null) {
            for (Neuron n : exclude) {
                idList.remove(n.getIdentifier());
            }
        }

        final List<Neuron> neuronList = new ArrayList<Neuron>();
        for (Long id : idList) {
            neuronList.add(getNeuron(id));
        }

        return neuronList;
    }

    
    private Long createNextId() {
        return nextId.getAndIncrement();
    }

    
    private void readObject(ObjectInputStream in) {
        throw new IllegalStateException();
    }

    
    private Object writeReplace() {
        final Neuron[] neuronList = neuronMap.values().toArray(new Neuron[0]);
        final long[][] neighbourIdList = new long[neuronList.length][];

        for (int i = 0; i < neuronList.length; i++) {
            final Collection<Neuron> neighbours = getNeighbours(neuronList[i]);
            final long[] neighboursId = new long[neighbours.size()];
            int count = 0;
            for (Neuron n : neighbours) {
                neighboursId[count] = n.getIdentifier();
                ++count;
            }
            neighbourIdList[i] = neighboursId;
        }

        return new SerializationProxy(nextId.get(),
                                      featureSize,
                                      neuronList,
                                      neighbourIdList);
    }

    
    private static class SerializationProxy implements Serializable {
        
        private static final long serialVersionUID = 20130207L;
        
        private final long nextId;
        
        private final int featureSize;
        
        private final Neuron[] neuronList;
        
        private final long[][] neighbourIdList;

        
        SerializationProxy(long nextId,
                           int featureSize,
                           Neuron[] neuronList,
                           long[][] neighbourIdList) {
            this.nextId = nextId;
            this.featureSize = featureSize;
            this.neuronList = neuronList;
            this.neighbourIdList = neighbourIdList;
        }

        
        private Object readResolve() {
            return new Network(nextId,
                               featureSize,
                               neuronList,
                               neighbourIdList);
        }
    }
}
