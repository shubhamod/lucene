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


public abstract class AbstractConvergenceChecker<PAIR>
    implements ConvergenceChecker<PAIR> {
    
    private final double relativeThreshold;
    
    private final double absoluteThreshold;

    
    public AbstractConvergenceChecker(final double relativeThreshold,
                                      final double absoluteThreshold) {
        this.relativeThreshold = relativeThreshold;
        this.absoluteThreshold = absoluteThreshold;
    }

    
    public double getRelativeThreshold() {
        return relativeThreshold;
    }

    
    public double getAbsoluteThreshold() {
        return absoluteThreshold;
    }

    
    public abstract boolean converged(int iteration,
                                      PAIR previous,
                                      PAIR current);
}
