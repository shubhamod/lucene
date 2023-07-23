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

import org.apache.lucene.util.hnsw.math.ml.neuralnet.sofm.util.ExponentialDecayFunction;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.sofm.util.QuasiSigmoidDecayFunction;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class NeighbourhoodSizeFunctionFactory {
    
    private NeighbourhoodSizeFunctionFactory() {}

    
    public static NeighbourhoodSizeFunction exponentialDecay(final double initValue,
                                                             final double valueAtNumCall,
                                                             final long numCall) {
        return new NeighbourhoodSizeFunction() {
            
            private final ExponentialDecayFunction decay
                = new ExponentialDecayFunction(initValue, valueAtNumCall, numCall);

            
            public int value(long n) {
                return (int) FastMath.rint(decay.value(n));
            }
        };
    }

    
    public static NeighbourhoodSizeFunction quasiSigmoidDecay(final double initValue,
                                                              final double slope,
                                                              final long numCall) {
        return new NeighbourhoodSizeFunction() {
            
            private final QuasiSigmoidDecayFunction decay
                = new QuasiSigmoidDecayFunction(initValue, slope, numCall);

            
            public int value(long n) {
                return (int) FastMath.rint(decay.value(n));
            }
        };
    }
}
