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
package org.apache.lucene.util.hnsw.math.optim;

import java.util.Arrays;


public class SimpleBounds implements OptimizationData {
    
    private final double[] lower;
    
    private final double[] upper;

    
    public SimpleBounds(double[] lB,
                        double[] uB) {
        lower = lB.clone();
        upper = uB.clone();
    }

    
    public double[] getLower() {
        return lower.clone();
    }
    
    public double[] getUpper() {
        return upper.clone();
    }

    
    public static SimpleBounds unbounded(int dim) {
        final double[] lB = new double[dim];
        Arrays.fill(lB, Double.NEGATIVE_INFINITY);
        final double[] uB = new double[dim];
        Arrays.fill(uB, Double.POSITIVE_INFINITY);

        return new SimpleBounds(lB, uB);
    }
}
