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

package org.apache.lucene.util.hnsw.math.ml.neuralnet.sofm.util;

import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class ExponentialDecayFunction {
    
    private final double a;
    
    private final double oneOverB;

    
    public ExponentialDecayFunction(double initValue,
                                    double valueAtNumCall,
                                    long numCall) {
        if (initValue <= 0) {
            throw new NotStrictlyPositiveException(initValue);
        }
        if (valueAtNumCall <= 0) {
            throw new NotStrictlyPositiveException(valueAtNumCall);
        }
        if (valueAtNumCall >= initValue) {
            throw new NumberIsTooLargeException(valueAtNumCall, initValue, false);
        }
        if (numCall <= 0) {
            throw new NotStrictlyPositiveException(numCall);
        }

        a = initValue;
        oneOverB = -FastMath.log(valueAtNumCall / initValue) / numCall;
    }

    
    public double value(long numCall) {
        return a * FastMath.exp(-numCall * oneOverB);
    }
}
