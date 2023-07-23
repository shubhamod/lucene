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

import org.apache.lucene.util.hnsw.math.exception.MathUnsupportedOperationException;


public class DefaultIterativeLinearSolverEvent extends IterativeLinearSolverEvent {

    
    private static final long serialVersionUID = 20120129L;

    
    private final RealVector b;

    
    private final RealVector r;

    
    private final double rnorm;

    
    private final RealVector x;

    
    public DefaultIterativeLinearSolverEvent(final Object source, final int iterations,
        final RealVector x, final RealVector b, final RealVector r,
        final double rnorm) {
        super(source, iterations);
        this.x = x;
        this.b = b;
        this.r = r;
        this.rnorm = rnorm;
    }

    
    public DefaultIterativeLinearSolverEvent(final Object source, final int iterations,
        final RealVector x, final RealVector b, final double rnorm) {
        super(source, iterations);
        this.x = x;
        this.b = b;
        this.r = null;
        this.rnorm = rnorm;
    }

    
    @Override
    public double getNormOfResidual() {
        return rnorm;
    }

    
    @Override
    public RealVector getResidual() {
        if (r != null) {
            return r;
        }
        throw new MathUnsupportedOperationException();
    }

    
    @Override
    public RealVector getRightHandSideVector() {
        return b;
    }

    
    @Override
    public RealVector getSolution() {
        return x;
    }

    
    @Override
    public boolean providesResidual() {
        return r != null;
    }
}
