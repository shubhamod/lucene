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
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;


public class LearningFactorFunctionFactory {
    
    private LearningFactorFunctionFactory() {}

    
    public static LearningFactorFunction exponentialDecay(final double initValue,
                                                          final double valueAtNumCall,
                                                          final long numCall) {
        if (initValue <= 0 ||
            initValue > 1) {
            throw new OutOfRangeException(initValue, 0, 1);
        }

        return new LearningFactorFunction() {
            
            private final ExponentialDecayFunction decay
                = new ExponentialDecayFunction(initValue, valueAtNumCall, numCall);

            
            public double value(long n) {
                return decay.value(n);
            }
        };
    }

    
    public static LearningFactorFunction quasiSigmoidDecay(final double initValue,
                                                           final double slope,
                                                           final long numCall) {
        if (initValue <= 0 ||
            initValue > 1) {
            throw new OutOfRangeException(initValue, 0, 1);
        }

        return new LearningFactorFunction() {
            
            private final QuasiSigmoidDecayFunction decay
                = new QuasiSigmoidDecayFunction(initValue, slope, numCall);

            
            public double value(long n) {
                return decay.value(n);
            }
        };
    }
}
