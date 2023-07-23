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

import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;


public class FixedGenerationCount implements StoppingCondition {
    
    private int numGenerations = 0;

    
    private final int maxGenerations;

    
    public FixedGenerationCount(final int maxGenerations) throws NumberIsTooSmallException {
        if (maxGenerations <= 0) {
            throw new NumberIsTooSmallException(maxGenerations, 1, true);
        }
        this.maxGenerations = maxGenerations;
    }

    
    public boolean isSatisfied(final Population population) {
        if (this.numGenerations < this.maxGenerations) {
            numGenerations++;
            return false;
        }
        return true;
    }

    
    public int getNumGenerations() {
        return numGenerations;
    }

}
