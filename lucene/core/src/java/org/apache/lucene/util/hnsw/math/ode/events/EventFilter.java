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

package org.apache.lucene.util.hnsw.math.ode.events;

import java.util.Arrays;



public class EventFilter implements EventHandler {

    
    private static final int HISTORY_SIZE = 100;

    
    private final EventHandler rawHandler;

    
    private final FilterType filter;

    
    private final Transformer[] transformers;

    
    private final double[] updates;

    
    private boolean forward;

    
    private double extremeT;

    
    public EventFilter(final EventHandler rawHandler, final FilterType filter) {
        this.rawHandler   = rawHandler;
        this.filter       = filter;
        this.transformers = new Transformer[HISTORY_SIZE];
        this.updates      = new double[HISTORY_SIZE];
    }

    
    public void init(double t0, double[] y0, double t) {

        // delegate to raw handler
        rawHandler.init(t0, y0, t);

        // initialize events triggering logic
        forward  = t >= t0;
        extremeT = forward ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        Arrays.fill(transformers, Transformer.UNINITIALIZED);
        Arrays.fill(updates, extremeT);

    }

    
    public double g(double t, double[] y) {

        final double rawG = rawHandler.g(t, y);

        // search which transformer should be applied to g
        if (forward) {
            final int last = transformers.length - 1;
            if (extremeT < t) {
                // we are at the forward end of the history

                // check if a new rough root has been crossed
                final Transformer previous = transformers[last];
                final Transformer next     = filter.selectTransformer(previous, rawG, forward);
                if (next != previous) {
                    // there is a root somewhere between extremeT and t.
                    // the new transformer is valid for t (this is how we have just computed
                    // it above), but it is in fact valid on both sides of the root, so
                    // it was already valid before t and even up to previous time. We store
                    // the switch at extremeT for safety, to ensure the previous transformer
                    // is not applied too close of the root
                    System.arraycopy(updates,      1, updates,      0, last);
                    System.arraycopy(transformers, 1, transformers, 0, last);
                    updates[last]      = extremeT;
                    transformers[last] = next;
                }

                extremeT = t;

                // apply the transform
                return next.transformed(rawG);

            } else {
                // we are in the middle of the history

                // select the transformer
                for (int i = last; i > 0; --i) {
                    if (updates[i] <= t) {
                        // apply the transform
                        return transformers[i].transformed(rawG);
                    }
                }

                return transformers[0].transformed(rawG);

            }
        } else {
            if (t < extremeT) {
                // we are at the backward end of the history

                // check if a new rough root has been crossed
                final Transformer previous = transformers[0];
                final Transformer next     = filter.selectTransformer(previous, rawG, forward);
                if (next != previous) {
                    // there is a root somewhere between extremeT and t.
                    // the new transformer is valid for t (this is how we have just computed
                    // it above), but it is in fact valid on both sides of the root, so
                    // it was already valid before t and even up to previous time. We store
                    // the switch at extremeT for safety, to ensure the previous transformer
                    // is not applied too close of the root
                    System.arraycopy(updates,      0, updates,      1, updates.length - 1);
                    System.arraycopy(transformers, 0, transformers, 1, transformers.length - 1);
                    updates[0]      = extremeT;
                    transformers[0] = next;
                }

                extremeT = t;

                // apply the transform
                return next.transformed(rawG);

            } else {
                // we are in the middle of the history

                // select the transformer
                for (int i = 0; i < updates.length - 1; ++i) {
                    if (t <= updates[i]) {
                        // apply the transform
                        return transformers[i].transformed(rawG);
                    }
                }

                return transformers[updates.length - 1].transformed(rawG);

            }
       }

    }

    
    public Action eventOccurred(double t, double[] y, boolean increasing) {
        // delegate to raw handler, fixing increasing status on the fly
        return rawHandler.eventOccurred(t, y, filter.getTriggeredIncreasing());
    }

    
    public void resetState(double t, double[] y) {
        // delegate to raw handler
        rawHandler.resetState(t, y);
    }

}
