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

import org.apache.lucene.util.hnsw.math.exception.MathInternalError;



public enum FilterType {

    
    TRIGGER_ONLY_DECREASING_EVENTS {

        
        @Override
        protected boolean getTriggeredIncreasing() {
            return false;
        }

        
        @Override
        protected  Transformer selectTransformer(final Transformer previous,
                                                 final double g, final boolean forward) {
            if (forward) {
                switch (previous) {
                    case UNINITIALIZED :
                        // we are initializing the first point
                        if (g > 0) {
                            // initialize as if previous root (i.e. backward one) was an ignored increasing event
                            return Transformer.MAX;
                        } else if (g < 0) {
                            // initialize as if previous root (i.e. backward one) was a triggered decreasing event
                            return Transformer.PLUS;
                        } else {
                            // we are exactly at a root, we don't know if it is an increasing
                            // or a decreasing event, we remain in uninitialized state
                            return Transformer.UNINITIALIZED;
                        }
                    case PLUS  :
                        if (g >= 0) {
                            // we have crossed the zero line on an ignored increasing event,
                            // we must change the transformer
                            return Transformer.MIN;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MINUS :
                        if (g >= 0) {
                            // we have crossed the zero line on an ignored increasing event,
                            // we must change the transformer
                            return Transformer.MAX;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MIN   :
                        if (g <= 0) {
                            // we have crossed the zero line on a triggered decreasing event,
                            // we must change the transformer
                            return Transformer.MINUS;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MAX   :
                        if (g <= 0) {
                            // we have crossed the zero line on a triggered decreasing event,
                            // we must change the transformer
                            return Transformer.PLUS;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    default    :
                        // this should never happen
                        throw new MathInternalError();
                }
            } else {
                switch (previous) {
                    case UNINITIALIZED :
                        // we are initializing the first point
                        if (g > 0) {
                            // initialize as if previous root (i.e. forward one) was a triggered decreasing event
                            return Transformer.MINUS;
                        } else if (g < 0) {
                            // initialize as if previous root (i.e. forward one) was an ignored increasing event
                            return Transformer.MIN;
                        } else {
                            // we are exactly at a root, we don't know if it is an increasing
                            // or a decreasing event, we remain in uninitialized state
                            return Transformer.UNINITIALIZED;
                        }
                    case PLUS  :
                        if (g <= 0) {
                            // we have crossed the zero line on an ignored increasing event,
                            // we must change the transformer
                            return Transformer.MAX;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MINUS :
                        if (g <= 0) {
                            // we have crossed the zero line on an ignored increasing event,
                            // we must change the transformer
                            return Transformer.MIN;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MIN   :
                        if (g >= 0) {
                            // we have crossed the zero line on a triggered decreasing event,
                            // we must change the transformer
                            return Transformer.PLUS;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MAX   :
                        if (g >= 0) {
                            // we have crossed the zero line on a triggered decreasing event,
                            // we must change the transformer
                            return Transformer.MINUS;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    default    :
                        // this should never happen
                        throw new MathInternalError();
                }
            }
        }

    },

    
    TRIGGER_ONLY_INCREASING_EVENTS {

        
        @Override
        protected boolean getTriggeredIncreasing() {
            return true;
        }

        
        @Override
        protected  Transformer selectTransformer(final Transformer previous,
                                                 final double g, final boolean forward) {
            if (forward) {
                switch (previous) {
                    case UNINITIALIZED :
                        // we are initializing the first point
                        if (g > 0) {
                            // initialize as if previous root (i.e. backward one) was a triggered increasing event
                            return Transformer.PLUS;
                        } else if (g < 0) {
                            // initialize as if previous root (i.e. backward one) was an ignored decreasing event
                            return Transformer.MIN;
                        } else {
                            // we are exactly at a root, we don't know if it is an increasing
                            // or a decreasing event, we remain in uninitialized state
                            return Transformer.UNINITIALIZED;
                        }
                    case PLUS  :
                        if (g <= 0) {
                            // we have crossed the zero line on an ignored decreasing event,
                            // we must change the transformer
                            return Transformer.MAX;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MINUS :
                        if (g <= 0) {
                            // we have crossed the zero line on an ignored decreasing event,
                            // we must change the transformer
                            return Transformer.MIN;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MIN   :
                        if (g >= 0) {
                            // we have crossed the zero line on a triggered increasing event,
                            // we must change the transformer
                            return Transformer.PLUS;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MAX   :
                        if (g >= 0) {
                            // we have crossed the zero line on a triggered increasing event,
                            // we must change the transformer
                            return Transformer.MINUS;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    default    :
                        // this should never happen
                        throw new MathInternalError();
                }
            } else {
                switch (previous) {
                    case UNINITIALIZED :
                        // we are initializing the first point
                        if (g > 0) {
                            // initialize as if previous root (i.e. forward one) was an ignored decreasing event
                            return Transformer.MAX;
                        } else if (g < 0) {
                            // initialize as if previous root (i.e. forward one) was a triggered increasing event
                            return Transformer.MINUS;
                        } else {
                            // we are exactly at a root, we don't know if it is an increasing
                            // or a decreasing event, we remain in uninitialized state
                            return Transformer.UNINITIALIZED;
                        }
                    case PLUS  :
                        if (g >= 0) {
                            // we have crossed the zero line on an ignored decreasing event,
                            // we must change the transformer
                            return Transformer.MIN;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MINUS :
                        if (g >= 0) {
                            // we have crossed the zero line on an ignored decreasing event,
                            // we must change the transformer
                            return Transformer.MAX;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MIN   :
                        if (g <= 0) {
                            // we have crossed the zero line on a triggered increasing event,
                            // we must change the transformer
                            return Transformer.MINUS;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    case MAX   :
                        if (g <= 0) {
                            // we have crossed the zero line on a triggered increasing event,
                            // we must change the transformer
                            return Transformer.PLUS;
                        } else {
                            // we are still in the same status
                            return previous;
                        }
                    default    :
                        // this should never happen
                        throw new MathInternalError();
                }
            }
        }

    };

    
    protected abstract boolean getTriggeredIncreasing();

    
    protected abstract Transformer selectTransformer(Transformer previous,
                                                     double g, boolean forward);

}
