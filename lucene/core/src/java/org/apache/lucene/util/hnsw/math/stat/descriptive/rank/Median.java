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
package org.apache.lucene.util.hnsw.math.stat.descriptive.rank;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.stat.ranking.NaNStrategy;
import org.apache.lucene.util.hnsw.math.util.KthSelector;



public class Median extends Percentile implements Serializable {

    
    private static final long serialVersionUID = -3961477041290915687L;

    
    private static final double FIXED_QUANTILE_50 = 50.0;

    
    public Median() {
        // No try-catch or advertised exception - arg is valid
        super(FIXED_QUANTILE_50);
    }

    
    public Median(Median original) throws NullArgumentException {
        super(original);
    }

    
    private Median(final EstimationType estimationType, final NaNStrategy nanStrategy,
                   final KthSelector kthSelector)
        throws MathIllegalArgumentException {
        super(FIXED_QUANTILE_50, estimationType, nanStrategy, kthSelector);
    }

    
    @Override
    public Median withEstimationType(final EstimationType newEstimationType) {
        return new Median(newEstimationType, getNaNStrategy(), getKthSelector());
    }

    
    @Override
    public Median withNaNStrategy(final NaNStrategy newNaNStrategy) {
        return new Median(getEstimationType(), newNaNStrategy, getKthSelector());
    }

    
    @Override
    public Median withKthSelector(final KthSelector newKthSelector) {
        return new Median(getEstimationType(), getNaNStrategy(), newKthSelector);
    }

}
