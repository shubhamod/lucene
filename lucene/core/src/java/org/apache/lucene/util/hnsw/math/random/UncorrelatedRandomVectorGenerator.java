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

package org.apache.lucene.util.hnsw.math.random;

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;



public class UncorrelatedRandomVectorGenerator
  implements RandomVectorGenerator {

    
    private final NormalizedRandomGenerator generator;

    
    private final double[] mean;

    
    private final double[] standardDeviation;

  
  public UncorrelatedRandomVectorGenerator(double[] mean,
                                           double[] standardDeviation,
                                           NormalizedRandomGenerator generator) {
    if (mean.length != standardDeviation.length) {
        throw new DimensionMismatchException(mean.length, standardDeviation.length);
    }
    this.mean              = mean.clone();
    this.standardDeviation = standardDeviation.clone();
    this.generator = generator;
  }

  
  public UncorrelatedRandomVectorGenerator(int dimension,
                                           NormalizedRandomGenerator generator) {
    mean              = new double[dimension];
    standardDeviation = new double[dimension];
    Arrays.fill(standardDeviation, 1.0);
    this.generator = generator;
  }

  
  public double[] nextVector() {

    double[] random = new double[mean.length];
    for (int i = 0; i < random.length; ++i) {
      random[i] = mean[i] + standardDeviation[i] * generator.nextNormalizedDouble();
    }

    return random;

  }

}
