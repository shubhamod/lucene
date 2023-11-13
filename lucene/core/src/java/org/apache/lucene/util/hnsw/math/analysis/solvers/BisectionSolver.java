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
package org.apache.lucene.util.hnsw.math.analysis.solvers;

import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;


public class BisectionSolver extends AbstractUnivariateSolver {
    
    private static final double DEFAULT_ABSOLUTE_ACCURACY = 1e-6;

    
    public BisectionSolver() {
        this(DEFAULT_ABSOLUTE_ACCURACY);
    }
    
    public BisectionSolver(double absoluteAccuracy) {
        super(absoluteAccuracy);
    }
    
    public BisectionSolver(double relativeAccuracy,
                           double absoluteAccuracy) {
        super(relativeAccuracy, absoluteAccuracy);
    }

    
    @Override
    protected double doSolve()
        throws TooManyEvaluationsException {
        double min = getMin();
        double max = getMax();
        verifyInterval(min, max);
        final double absoluteAccuracy = getAbsoluteAccuracy();
        double m;
        double fm;
        double fmin;

        while (true) {
            m = UnivariateSolverUtils.midpoint(min, max);
            fmin = computeObjectiveValue(min);
            fm = computeObjectiveValue(m);

            if (fm * fmin > 0) {
                // max and m bracket the root.
                min = m;
            } else {
                // min and m bracket the root.
                max = m;
            }

            if (FastMath.abs(max - min) <= absoluteAccuracy) {
                m = UnivariateSolverUtils.midpoint(min, max);
                return m;
            }
        }
    }
}
