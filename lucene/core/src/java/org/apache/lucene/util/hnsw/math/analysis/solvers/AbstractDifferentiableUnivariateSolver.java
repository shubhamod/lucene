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
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;


@Deprecated
public abstract class AbstractDifferentiableUnivariateSolver
    extends BaseAbstractUnivariateSolver<DifferentiableUnivariateFunction>
    implements DifferentiableUnivariateSolver {
    
    private UnivariateFunction functionDerivative;

    
    protected AbstractDifferentiableUnivariateSolver(final double absoluteAccuracy) {
        super(absoluteAccuracy);
    }

    
    protected AbstractDifferentiableUnivariateSolver(final double relativeAccuracy,
                                                     final double absoluteAccuracy,
                                                     final double functionValueAccuracy) {
        super(relativeAccuracy, absoluteAccuracy, functionValueAccuracy);
    }

    
    protected double computeDerivativeObjectiveValue(double point)
        throws TooManyEvaluationsException {
        incrementEvaluationCount();
        return functionDerivative.value(point);
    }

    
    @Override
    protected void setup(int maxEval, DifferentiableUnivariateFunction f,
                         double min, double max, double startValue) {
        super.setup(maxEval, f, min, max, startValue);
        functionDerivative = f.derivative();
    }
}
