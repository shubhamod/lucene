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

import org.apache.lucene.util.hnsw.math.analysis.DifferentiableUnivariateFunction;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;


@Deprecated
public class NewtonSolver extends AbstractDifferentiableUnivariateSolver {
    
    private static final double DEFAULT_ABSOLUTE_ACCURACY = 1e-6;

    
    public NewtonSolver() {
        this(DEFAULT_ABSOLUTE_ACCURACY);
    }
    
    public NewtonSolver(double absoluteAccuracy) {
        super(absoluteAccuracy);
    }

    
    @Override
    public double solve(int maxEval, final DifferentiableUnivariateFunction f,
                        final double min, final double max)
        throws TooManyEvaluationsException {
        return super.solve(maxEval, f, UnivariateSolverUtils.midpoint(min, max));
    }

    
    @Override
    protected double doSolve()
        throws TooManyEvaluationsException {
        final double startValue = getStartValue();
        final double absoluteAccuracy = getAbsoluteAccuracy();

        double x0 = startValue;
        double x1;
        while (true) {
            x1 = x0 - (computeObjectiveValue(x0) / computeDerivativeObjectiveValue(x0));
            if (FastMath.abs(x1 - x0) <= absoluteAccuracy) {
                return x1;
            }

            x0 = x1;
        }
    }
}
