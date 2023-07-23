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
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.util.Precision;



public class Neuron implements Serializable {
    
    private static final long serialVersionUID = 20130207L;
    
    private final long identifier;
    
    private final int size;
    
    private final AtomicReference<double[]> features;
    
    private final AtomicLong numberOfAttemptedUpdates = new AtomicLong(0);
    
    private final AtomicLong numberOfSuccessfulUpdates = new AtomicLong(0);

    
    Neuron(long identifier,
           double[] features) {
        this.identifier = identifier;
        this.size = features.length;
        this.features = new AtomicReference<double[]>(features.clone());
    }

    
    public synchronized Neuron copy() {
        final Neuron copy = new Neuron(getIdentifier(),
                                       getFeatures());
        copy.numberOfAttemptedUpdates.set(numberOfAttemptedUpdates.get());
        copy.numberOfSuccessfulUpdates.set(numberOfSuccessfulUpdates.get());

        return copy;
    }

    
    public long getIdentifier() {
        return identifier;
    }

    
    public int getSize() {
        return size;
    }

    
    public double[] getFeatures() {
        return features.get().clone();
    }

    
    public boolean compareAndSetFeatures(double[] expect,
                                         double[] update) {
        if (update.length != size) {
            throw new DimensionMismatchException(update.length, size);
        }

        // Get the internal reference. Note that this must not be a copy;
        // otherwise the "compareAndSet" below will always fail.
        final double[] current = features.get();
        if (!containSameValues(current, expect)) {
            // Some other thread already modified the state.
            return false;
        }

        // Increment attempt counter.
        numberOfAttemptedUpdates.incrementAndGet();

        if (features.compareAndSet(current, update.clone())) {
            // The current thread could atomically update the state (attempt succeeded).
            numberOfSuccessfulUpdates.incrementAndGet();
            return true;
        } else {
            // Some other thread came first (attempt failed).
            return false;
        }
    }

    
    public long getNumberOfAttemptedUpdates() {
        return numberOfAttemptedUpdates.get();
    }

    
    public long getNumberOfSuccessfulUpdates() {
        return numberOfSuccessfulUpdates.get();
    }

    
    private boolean containSameValues(double[] current,
                                      double[] expect) {
        if (expect.length != size) {
            throw new DimensionMismatchException(expect.length, size);
        }

        for (int i = 0; i < size; i++) {
            if (!Precision.equals(current[i], expect[i])) {
                return false;
            }
        }
        return true;
    }

    
    private void readObject(ObjectInputStream in) {
        throw new IllegalStateException();
    }

    
    private Object writeReplace() {
        return new SerializationProxy(identifier,
                                      features.get());
    }

    
    private static class SerializationProxy implements Serializable {
        
        private static final long serialVersionUID = 20130207L;
        
        private final double[] features;
        
        private final long identifier;

        
        SerializationProxy(long identifier,
                           double[] features) {
            this.identifier = identifier;
            this.features = features;
        }

        
        private Object readResolve() {
            return new Neuron(identifier,
                              features);
        }
    }
}
