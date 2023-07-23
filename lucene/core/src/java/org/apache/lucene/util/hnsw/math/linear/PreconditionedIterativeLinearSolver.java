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


public abstract class PreconditionedIterativeLinearSolver
    extends IterativeLinearSolver {

    
    public PreconditionedIterativeLinearSolver(final int maxIterations) {
        super(maxIterations);
    }

    
    public PreconditionedIterativeLinearSolver(final IterationManager manager)
        throws NullArgumentException {
        super(manager);
    }

    
    public RealVector solve(final RealLinearOperator a,
        final RealLinearOperator m, final RealVector b, final RealVector x0)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException {
        MathUtils.checkNotNull(x0);
        return solveInPlace(a, m, b, x0.copy());
    }

    
    @Override
    public RealVector solve(final RealLinearOperator a, final RealVector b)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException {
        MathUtils.checkNotNull(a);
        final RealVector x = new ArrayRealVector(a.getColumnDimension());
        x.set(0.);
        return solveInPlace(a, null, b, x);
    }

    
    @Override
    public RealVector solve(final RealLinearOperator a, final RealVector b,
                            final RealVector x0)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException {
        MathUtils.checkNotNull(x0);
        return solveInPlace(a, null, b, x0.copy());
    }

    
    protected static void checkParameters(final RealLinearOperator a,
        final RealLinearOperator m, final RealVector b, final RealVector x0)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException {
        checkParameters(a, b, x0);
        if (m != null) {
            if (m.getColumnDimension() != m.getRowDimension()) {
                throw new NonSquareOperatorException(m.getColumnDimension(),
                                                     m.getRowDimension());
            }
            if (m.getRowDimension() != a.getRowDimension()) {
                throw new DimensionMismatchException(m.getRowDimension(),
                                                     a.getRowDimension());
            }
        }
    }

    
    public RealVector solve(RealLinearOperator a, RealLinearOperator m,
        RealVector b) throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException {
        MathUtils.checkNotNull(a);
        final RealVector x = new ArrayRealVector(a.getColumnDimension());
        return solveInPlace(a, m, b, x);
    }

    
    public abstract RealVector solveInPlace(RealLinearOperator a,
        RealLinearOperator m, RealVector b, RealVector x0) throws
        NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException;

    
    @Override
    public RealVector solveInPlace(final RealLinearOperator a,
        final RealVector b, final RealVector x0) throws
        NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException {
        return solveInPlace(a, null, b, x0);
    }
}
