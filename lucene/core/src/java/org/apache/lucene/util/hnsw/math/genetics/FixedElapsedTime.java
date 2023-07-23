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
package org.apache.lucene.util.hnsw.math.genetics;

import java.util.concurrent.TimeUnit;

import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;


public class FixedElapsedTime implements StoppingCondition {
    
    private final long maxTimePeriod;

    
    private long endTime = -1;

    
    public FixedElapsedTime(final long maxTime) throws NumberIsTooSmallException {
        this(maxTime, TimeUnit.SECONDS);
    }

    
    public FixedElapsedTime(final long maxTime, final TimeUnit unit) throws NumberIsTooSmallException {
        if (maxTime < 0) {
            throw new NumberIsTooSmallException(maxTime, 0, true);
        }
        maxTimePeriod = unit.toNanos(maxTime);
    }

    
    public boolean isSatisfied(final Population population) {
        if (endTime < 0) {
            endTime = System.nanoTime() + maxTimePeriod;
        }

        return System.nanoTime() >= endTime;
    }
}
