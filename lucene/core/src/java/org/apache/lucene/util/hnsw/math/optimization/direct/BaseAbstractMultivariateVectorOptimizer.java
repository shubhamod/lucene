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

package org.apache.lucene.util.hnsw.math.optimization.direct;

import org.apache.lucene.util.hnsw.math.util.Incrementor;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.optimization.OptimizationData;
import org.apache.lucene.util.hnsw.math.optimization.InitialGuess;
import org.apache.lucene.util.hnsw.math.optimization.Target;
import org.apache.lucene.util.hnsw.math.optimization.Weight;
import org.apache.lucene.util.hnsw.math.optimization.BaseMultivariateVectorOptimizer;
import org.apache.lucene.util.hnsw.math.optimization.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optimization.PointVectorValuePair;
import org.apache.lucene.util.hnsw.math.optimization.SimpleVectorValueChecker;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;


@Deprecated
public abstract class BaseAbstractMultivariateVectorOptimizer<FUNC extends MultivariateVectorFunction>
    implements BaseMultivariateVectorOptimizer<FUNC> {
    
    protected final Incrementor evaluations = new Incrementor();
    
    private ConvergenceChecker<PointVectorValuePair> checker;
    
    private double[] target;
    
    private RealMatrix weightMatrix;
    
    @Deprecated
    private double[] weight;
    
    private double[] start;
    
    private FUNC function;

    
    @Deprecated
    protected BaseAbstractMultivariateVectorOptimizer() {
        this(new SimpleVectorValueChecker());
    }
    
    protected BaseAbstractMultivariateVectorOptimizer(ConvergenceChecker<PointVectorValuePair> checker) {
        this.checker = checker;
    }

    
    public int getMaxEvaluations() {
        return evaluations.getMaximalCount();
    }

    
    public int getEvaluations() {
        return evaluations.getCount();
    }

    
    public ConvergenceChecker<PointVectorValuePair> getConvergenceChecker() {
        return checker;
    }

    
    protected double[] computeObjectiveValue(double[] point) {
        try {
            evaluations.incrementCount();
        } catch (MaxCountExceededException e) {
            throw new TooManyEvaluationsException(e.getMax());
        }
        return function.value(point);
    }

    
    @Deprecated
    public PointVectorValuePair optimize(int maxEval, FUNC f, double[] t, double[] w,
                                         double[] startPoint) {
        return optimizeInternal(maxEval, f, t, w, startPoint);
    }

    
    protected PointVectorValuePair optimize(int maxEval,
                                            FUNC f,
                                            OptimizationData... optData)
        throws TooManyEvaluationsException,
               DimensionMismatchException {
        return optimizeInternal(maxEval, f, optData);
    }

    
    @Deprecated
    protected PointVectorValuePair optimizeInternal(final int maxEval, final FUNC f,
                                                    final double[] t, final double[] w,
                                                    final double[] startPoint) {
        // Checks.
        if (f == null) {
            throw new NullArgumentException();
        }
        if (t == null) {
            throw new NullArgumentException();
        }
        if (w == null) {
            throw new NullArgumentException();
        }
        if (startPoint == null) {
            throw new NullArgumentException();
        }
        if (t.length != w.length) {
            throw new DimensionMismatchException(t.length, w.length);
        }

        return optimizeInternal(maxEval, f,
                                new Target(t),
                                new Weight(w),
                                new InitialGuess(startPoint));
    }

    
    protected PointVectorValuePair optimizeInternal(int maxEval,
                                                    FUNC f,
                                                    OptimizationData... optData)
        throws TooManyEvaluationsException,
               DimensionMismatchException {
        // Set internal state.
        evaluations.setMaximalCount(maxEval);
        evaluations.resetCount();
        function = f;
        // Retrieve other settings.
        parseOptimizationData(optData);
        // Check input consistency.
        checkParameters();
        // Allow subclasses to reset their own internal state.
        setUp();
        // Perform computation.
        return doOptimize();
    }

    
    public double[] getStartPoint() {
        return start.clone();
    }

    
    public RealMatrix getWeight() {
        return weightMatrix.copy();
    }
    
    public double[] getTarget() {
        return target.clone();
    }

    
    protected FUNC getObjectiveFunction() {
        return function;
    }

    
    protected abstract PointVectorValuePair doOptimize();

    
    @Deprecated
    protected double[] getTargetRef() {
        return target;
    }
    
    @Deprecated
    protected double[] getWeightRef() {
        return weight;
    }

    
    protected void setUp() {
        // XXX Temporary code until the new internal data is used everywhere.
        final int dim = target.length;
        weight = new double[dim];
        for (int i = 0; i < dim; i++) {
            weight[i] = weightMatrix.getEntry(i, i);
        }
    }

    
    private void parseOptimizationData(OptimizationData... optData) {
        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof Target) {
                target = ((Target) data).getTarget();
                continue;
            }
            if (data instanceof Weight) {
                weightMatrix = ((Weight) data).getWeight();
                continue;
            }
            if (data instanceof InitialGuess) {
                start = ((InitialGuess) data).getInitialGuess();
                continue;
            }
        }
    }

    
    private void checkParameters() {
        if (target.length != weightMatrix.getColumnDimension()) {
            throw new DimensionMismatchException(target.length,
                                                 weightMatrix.getColumnDimension());
        }
    }
}
