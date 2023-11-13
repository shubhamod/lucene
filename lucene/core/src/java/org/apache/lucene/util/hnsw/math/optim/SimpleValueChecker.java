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

import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;


public class SimpleValueChecker
    extends AbstractConvergenceChecker<PointValuePair> {
    
    private static final int ITERATION_CHECK_DISABLED = -1;
    
    private final int maxIterationCount;

    
    public SimpleValueChecker(final double relativeThreshold,
                              final double absoluteThreshold) {
        super(relativeThreshold, absoluteThreshold);
        maxIterationCount = ITERATION_CHECK_DISABLED;
    }

    
    public SimpleValueChecker(final double relativeThreshold,
                              final double absoluteThreshold,
                              final int maxIter) {
        super(relativeThreshold, absoluteThreshold);

        if (maxIter <= 0) {
            throw new NotStrictlyPositiveException(maxIter);
        }
        maxIterationCount = maxIter;
    }

    
    @Override
    public boolean converged(final int iteration,
                             final PointValuePair previous,
                             final PointValuePair current) {
        if (maxIterationCount != ITERATION_CHECK_DISABLED && iteration >= maxIterationCount) {
            return true;
        }

        final double p = previous.getValue();
        final double c = current.getValue();
        final double difference = FastMath.abs(p - c);
        final double size = FastMath.max(FastMath.abs(p), FastMath.abs(c));
        return difference <= size * getRelativeThreshold() ||
            difference <= getAbsoluteThreshold();
    }
}
