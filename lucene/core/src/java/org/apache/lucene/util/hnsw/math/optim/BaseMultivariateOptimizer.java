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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;


public abstract class BaseMultivariateOptimizer<PAIR>
    extends BaseOptimizer<PAIR> {
    
    private double[] start;
    
    private double[] lowerBound;
    
    private double[] upperBound;

    
    protected BaseMultivariateOptimizer(ConvergenceChecker<PAIR> checker) {
        super(checker);
    }

    
    @Override
    public PAIR optimize(OptimizationData... optData) {
        // Perform optimization.
        return super.optimize(optData);
    }

    
    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        // Allow base class to register its own data.
        super.parseOptimizationData(optData);

        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof InitialGuess) {
                start = ((InitialGuess) data).getInitialGuess();
                continue;
            }
            if (data instanceof SimpleBounds) {
                final SimpleBounds bounds = (SimpleBounds) data;
                lowerBound = bounds.getLower();
                upperBound = bounds.getUpper();
                continue;
            }
        }

        // Check input consistency.
        checkParameters();
    }

    
    public double[] getStartPoint() {
        return start == null ? null : start.clone();
    }
    
    public double[] getLowerBound() {
        return lowerBound == null ? null : lowerBound.clone();
    }
    
    public double[] getUpperBound() {
        return upperBound == null ? null : upperBound.clone();
    }

    
    private void checkParameters() {
        if (start != null) {
            final int dim = start.length;
            if (lowerBound != null) {
                if (lowerBound.length != dim) {
                    throw new DimensionMismatchException(lowerBound.length, dim);
                }
                for (int i = 0; i < dim; i++) {
                    final double v = start[i];
                    final double lo = lowerBound[i];
                    if (v < lo) {
                        throw new NumberIsTooSmallException(v, lo, true);
                    }
                }
            }
            if (upperBound != null) {
                if (upperBound.length != dim) {
                    throw new DimensionMismatchException(upperBound.length, dim);
                }
                for (int i = 0; i < dim; i++) {
                    final double v = start[i];
                    final double hi = upperBound[i];
                    if (v > hi) {
                        throw new NumberIsTooLargeException(v, hi, true);
                    }
                }
            }
        }
    }
}
