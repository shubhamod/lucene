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
package org.apache.lucene.util.hnsw.math.dfp;


import org.apache.lucene.util.hnsw.math.analysis.RealFieldUnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.solvers.AllowedSolution;
import org.apache.lucene.util.hnsw.math.analysis.solvers.FieldBracketingNthOrderBrentSolver;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


@Deprecated
public class BracketingNthOrderBrentSolverDFP extends FieldBracketingNthOrderBrentSolver<Dfp> {

    
    public BracketingNthOrderBrentSolverDFP(final Dfp relativeAccuracy,
                                            final Dfp absoluteAccuracy,
                                            final Dfp functionValueAccuracy,
                                            final int maximalOrder)
        throws NumberIsTooSmallException {
        super(relativeAccuracy, absoluteAccuracy, functionValueAccuracy, maximalOrder);
    }

    
    @Override
    public Dfp getAbsoluteAccuracy() {
        return super.getAbsoluteAccuracy();
    }

    
    @Override
    public Dfp getRelativeAccuracy() {
        return super.getRelativeAccuracy();
    }

    
    @Override
    public Dfp getFunctionValueAccuracy() {
        return super.getFunctionValueAccuracy();
    }

    
    public Dfp solve(final int maxEval, final UnivariateDfpFunction f,
                     final Dfp min, final Dfp max, final AllowedSolution allowedSolution)
        throws NullArgumentException, NoBracketingException {
        return solve(maxEval, f, min, max, min.add(max).divide(2), allowedSolution);
    }

    
    public Dfp solve(final int maxEval, final UnivariateDfpFunction f,
                     final Dfp min, final Dfp max, final Dfp startValue,
                     final AllowedSolution allowedSolution)
        throws NullArgumentException, NoBracketingException {

        // checks
        MathUtils.checkNotNull(f);

        // wrap the function
        RealFieldUnivariateFunction<Dfp> fieldF = new RealFieldUnivariateFunction<Dfp>() {

            
            public Dfp value(final Dfp x) {
                return f.value(x);
            }
        };

        // delegate to general field solver
        return solve(maxEval, fieldF, min, max, startValue, allowedSolution);

    }

}
