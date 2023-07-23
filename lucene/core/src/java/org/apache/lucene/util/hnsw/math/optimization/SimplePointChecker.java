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

package org.apache.lucene.util.hnsw.math.optimization;

import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Pair;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;


@Deprecated
public class SimplePointChecker<PAIR extends Pair<double[], ? extends Object>>
    extends AbstractConvergenceChecker<PAIR> {
    
    private static final int ITERATION_CHECK_DISABLED = -1;
    
    private final int maxIterationCount;

    
    @Deprecated
    public SimplePointChecker() {
        maxIterationCount = ITERATION_CHECK_DISABLED;
    }

    
    public SimplePointChecker(final double relativeThreshold,
                              final double absoluteThreshold) {
        super(relativeThreshold, absoluteThreshold);
        maxIterationCount = ITERATION_CHECK_DISABLED;
    }

    
    public SimplePointChecker(final double relativeThreshold,
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
                             final PAIR previous,
                             final PAIR current) {
        if (maxIterationCount != ITERATION_CHECK_DISABLED && iteration >= maxIterationCount) {
            return true;
        }

        final double[] p = previous.getKey();
        final double[] c = current.getKey();
        for (int i = 0; i < p.length; ++i) {
            final double pi = p[i];
            final double ci = c[i];
            final double difference = FastMath.abs(pi - ci);
            final double size = FastMath.max(FastMath.abs(pi), FastMath.abs(ci));
            if (difference > size * getRelativeThreshold() &&
                difference > getAbsoluteThreshold()) {
                return false;
            }
        }
        return true;
    }
}
