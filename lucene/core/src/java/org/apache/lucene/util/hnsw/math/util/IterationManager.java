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
package org.apache.lucene.util.hnsw.math.util;

import java.util.Collection;
import java.util.concurrent.CopyOnWriteArrayList;

import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;


public class IterationManager {

    
    private IntegerSequence.Incrementor iterations;

    
    private final Collection<IterationListener> listeners;

    
    public IterationManager(final int maxIterations) {
        this.iterations = IntegerSequence.Incrementor.create().withMaximalCount(maxIterations);
        this.listeners = new CopyOnWriteArrayList<IterationListener>();
    }

    
    @Deprecated
    public IterationManager(final int maxIterations,
                            final Incrementor.MaxCountExceededCallback callBack) {
        this(maxIterations, new IntegerSequence.Incrementor.MaxCountExceededCallback() {
            
            public void trigger(final int maximalCount) throws MaxCountExceededException {
                callBack.trigger(maximalCount);
            }
        });
    }

    
    public IterationManager(final int maxIterations,
                            final IntegerSequence.Incrementor.MaxCountExceededCallback callBack) {
        this.iterations = IntegerSequence.Incrementor.create().withMaximalCount(maxIterations).withCallback(callBack);
        this.listeners = new CopyOnWriteArrayList<IterationListener>();
    }

    
    public void addIterationListener(final IterationListener listener) {
        listeners.add(listener);
    }

    
    public void fireInitializationEvent(final IterationEvent e) {
        for (IterationListener l : listeners) {
            l.initializationPerformed(e);
        }
    }

    
    public void fireIterationPerformedEvent(final IterationEvent e) {
        for (IterationListener l : listeners) {
            l.iterationPerformed(e);
        }
    }

    
    public void fireIterationStartedEvent(final IterationEvent e) {
        for (IterationListener l : listeners) {
            l.iterationStarted(e);
        }
    }

    
    public void fireTerminationEvent(final IterationEvent e) {
        for (IterationListener l : listeners) {
            l.terminationPerformed(e);
        }
    }

    
    public int getIterations() {
        return iterations.getCount();
    }

    
    public int getMaxIterations() {
        return iterations.getMaximalCount();
    }

    
    public void incrementIterationCount()
        throws MaxCountExceededException {
        iterations.increment();
    }

    
    public void removeIterationListener(final IterationListener listener) {
        listeners.remove(listener);
    }

    
    public void resetIterationCount() {
        iterations = iterations.withStart(0);
    }
}
