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
package org.apache.lucene.util.hnsw.math.linear;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.util.IterationManager;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public abstract class IterativeLinearSolver {

    
    private final IterationManager manager;

    
    public IterativeLinearSolver(final int maxIterations) {
        this.manager = new IterationManager(maxIterations);
    }

    
    public IterativeLinearSolver(final IterationManager manager)
        throws NullArgumentException {
        MathUtils.checkNotNull(manager);
        this.manager = manager;
    }

    
    protected static void checkParameters(final RealLinearOperator a,
        final RealVector b, final RealVector x0) throws
        NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException {
        MathUtils.checkNotNull(a);
        MathUtils.checkNotNull(b);
        MathUtils.checkNotNull(x0);
        if (a.getRowDimension() != a.getColumnDimension()) {
            throw new NonSquareOperatorException(a.getRowDimension(),
                                                       a.getColumnDimension());
        }
        if (b.getDimension() != a.getRowDimension()) {
            throw new DimensionMismatchException(b.getDimension(),
                                                 a.getRowDimension());
        }
        if (x0.getDimension() != a.getColumnDimension()) {
            throw new DimensionMismatchException(x0.getDimension(),
                                                 a.getColumnDimension());
        }
    }

    
    public IterationManager getIterationManager() {
        return manager;
    }

    
    public RealVector solve(final RealLinearOperator a, final RealVector b)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException {
        MathUtils.checkNotNull(a);
        final RealVector x = new ArrayRealVector(a.getColumnDimension());
        x.set(0.);
        return solveInPlace(a, b, x);
    }

    
    public RealVector solve(RealLinearOperator a, RealVector b, RealVector x0)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException {
        MathUtils.checkNotNull(x0);
        return solveInPlace(a, b, x0.copy());
    }

    
    public abstract RealVector solveInPlace(RealLinearOperator a, RealVector b,
        RealVector x0) throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException;
}
