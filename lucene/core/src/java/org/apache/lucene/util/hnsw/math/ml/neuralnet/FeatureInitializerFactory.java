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

import org.apache.lucene.util.hnsw.math.distribution.RealDistribution;
import org.apache.lucene.util.hnsw.math.distribution.UniformRealDistribution;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.function.Constant;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;


public class FeatureInitializerFactory {
    
    private FeatureInitializerFactory() {}

    
    public static FeatureInitializer uniform(final RandomGenerator rng,
                                             final double min,
                                             final double max) {
        return randomize(new UniformRealDistribution(rng, min, max),
                         function(new Constant(0), 0, 0));
    }

    
    public static FeatureInitializer uniform(final double min,
                                             final double max) {
        return randomize(new UniformRealDistribution(min, max),
                         function(new Constant(0), 0, 0));
    }

    
    public static FeatureInitializer function(final UnivariateFunction f,
                                              final double init,
                                              final double inc) {
        return new FeatureInitializer() {
            
            private double arg = init;

            
            public double value() {
                final double result = f.value(arg);
                arg += inc;
                return result;
            }
        };
    }

    
    public static FeatureInitializer randomize(final RealDistribution random,
                                               final FeatureInitializer orig) {
        return new FeatureInitializer() {
            
            public double value() {
                return orig.value() + random.sample();
            }
        };
    }
}
