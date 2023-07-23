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

package org.apache.lucene.util.hnsw.math.ode.nonstiff;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.ode.FieldEquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.FieldExpandableODE;
import org.apache.lucene.util.hnsw.math.ode.FieldODEState;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;



public abstract class EmbeddedRungeKuttaFieldIntegrator<T extends RealFieldElement<T>>
    extends AdaptiveStepsizeFieldIntegrator<T>
    implements FieldButcherArrayProvider<T> {

    
    private final int fsal;

    
    private final T[] c;

    
    private final T[][] a;

    
    private final T[] b;

    
    private final T exp;

    
    private T safety;

    
    private T minReduction;

    
    private T maxGrowth;

    
    protected EmbeddedRungeKuttaFieldIntegrator(final Field<T> field, final String name, final int fsal,
                                                final double minStep, final double maxStep,
                                                final double scalAbsoluteTolerance,
                                                final double scalRelativeTolerance) {

        super(field, name, minStep, maxStep, scalAbsoluteTolerance, scalRelativeTolerance);

        this.fsal = fsal;
        this.c    = getC();
        this.a    = getA();
        this.b    = getB();

        exp = field.getOne().divide(-getOrder());

        // set the default values of the algorithm control parameters
        setSafety(field.getZero().add(0.9));
        setMinReduction(field.getZero().add(0.2));
        setMaxGrowth(field.getZero().add(10.0));

    }

    
    protected EmbeddedRungeKuttaFieldIntegrator(final Field<T> field, final String name, final int fsal,
                                                final double   minStep, final double maxStep,
                                                final double[] vecAbsoluteTolerance,
                                                final double[] vecRelativeTolerance) {

        super(field, name, minStep, maxStep, vecAbsoluteTolerance, vecRelativeTolerance);

        this.fsal = fsal;
        this.c    = getC();
        this.a    = getA();
        this.b    = getB();

        exp = field.getOne().divide(-getOrder());

        // set the default values of the algorithm control parameters
        setSafety(field.getZero().add(0.9));
        setMinReduction(field.getZero().add(0.2));
        setMaxGrowth(field.getZero().add(10.0));

    }

    
    protected T fraction(final int p, final int q) {
        return getField().getOne().multiply(p).divide(q);
    }

    
    protected T fraction(final double p, final double q) {
        return getField().getOne().multiply(p).divide(q);
    }

    
    protected abstract RungeKuttaFieldStepInterpolator<T> createInterpolator(boolean forward, T[][] yDotK,
                                                                             final FieldODEStateAndDerivative<T> globalPreviousState,
                                                                             final FieldODEStateAndDerivative<T> globalCurrentState,
                                                                             FieldEquationsMapper<T> mapper);
    
    public abstract int getOrder();

    
    public T getSafety() {
        return safety;
    }

    
    public void setSafety(final T safety) {
        this.safety = safety;
    }

    
    public FieldODEStateAndDerivative<T> integrate(final FieldExpandableODE<T> equations,
                                                   final FieldODEState<T> initialState, final T finalTime)
        throws NumberIsTooSmallException, DimensionMismatchException,
        MaxCountExceededException, NoBracketingException {

        sanityChecks(initialState, finalTime);
        final T   t0 = initialState.getTime();
        final T[] y0 = equations.getMapper().mapState(initialState);
        setStepStart(initIntegration(equations, t0, y0, finalTime));
        final boolean forward = finalTime.subtract(initialState.getTime()).getReal() > 0;

        // create some internal working arrays
        final int   stages = c.length + 1;
        T[]         y      = y0;
        final T[][] yDotK  = MathArrays.buildArray(getField(), stages, -1);
        final T[]   yTmp   = MathArrays.buildArray(getField(), y0.length);

        // set up integration control objects
        T  hNew           = getField().getZero();
        boolean firstTime = true;

        // main integration loop
        setIsLastStep(false);
        do {

            // iterate over step size, ensuring local normalized error is smaller than 1
            T error = getField().getZero().add(10);
            while (error.subtract(1.0).getReal() >= 0) {

                // first stage
                y        = equations.getMapper().mapState(getStepStart());
                yDotK[0] = equations.getMapper().mapDerivative(getStepStart());

                if (firstTime) {
                    final T[] scale = MathArrays.buildArray(getField(), mainSetDimension);
                    if (vecAbsoluteTolerance == null) {
                        for (int i = 0; i < scale.length; ++i) {
                            scale[i] = y[i].abs().multiply(scalRelativeTolerance).add(scalAbsoluteTolerance);
                        }
                    } else {
                        for (int i = 0; i < scale.length; ++i) {
                            scale[i] = y[i].abs().multiply(vecRelativeTolerance[i]).add(vecAbsoluteTolerance[i]);
                        }
                    }
                    hNew = initializeStep(forward, getOrder(), scale, getStepStart(), equations.getMapper());
                    firstTime = false;
                }

                setStepSize(hNew);
                if (forward) {
                    if (getStepStart().getTime().add(getStepSize()).subtract(finalTime).getReal() >= 0) {
                        setStepSize(finalTime.subtract(getStepStart().getTime()));
                    }
                } else {
                    if (getStepStart().getTime().add(getStepSize()).subtract(finalTime).getReal() <= 0) {
                        setStepSize(finalTime.subtract(getStepStart().getTime()));
                    }
                }

                // next stages
                for (int k = 1; k < stages; ++k) {

                    for (int j = 0; j < y0.length; ++j) {
                        T sum = yDotK[0][j].multiply(a[k-1][0]);
                        for (int l = 1; l < k; ++l) {
                            sum = sum.add(yDotK[l][j].multiply(a[k-1][l]));
                        }
                        yTmp[j] = y[j].add(getStepSize().multiply(sum));
                    }

                    yDotK[k] = computeDerivatives(getStepStart().getTime().add(getStepSize().multiply(c[k-1])), yTmp);

                }

                // estimate the state at the end of the step
                for (int j = 0; j < y0.length; ++j) {
                    T sum    = yDotK[0][j].multiply(b[0]);
                    for (int l = 1; l < stages; ++l) {
                        sum = sum.add(yDotK[l][j].multiply(b[l]));
                    }
                    yTmp[j] = y[j].add(getStepSize().multiply(sum));
                }

                // estimate the error at the end of the step
                error = estimateError(yDotK, y, yTmp, getStepSize());
                if (error.subtract(1.0).getReal() >= 0) {
                    // reject the step and attempt to reduce error by stepsize control
                    final T factor = MathUtils.min(maxGrowth,
                                                   MathUtils.max(minReduction, safety.multiply(error.pow(exp))));
                    hNew = filterStep(getStepSize().multiply(factor), forward, false);
                }

            }
            final T   stepEnd = getStepStart().getTime().add(getStepSize());
            final T[] yDotTmp = (fsal >= 0) ? yDotK[fsal] : computeDerivatives(stepEnd, yTmp);
            final FieldODEStateAndDerivative<T> stateTmp = new FieldODEStateAndDerivative<T>(stepEnd, yTmp, yDotTmp);

            // local error is small enough: accept the step, trigger events and step handlers
            System.arraycopy(yTmp, 0, y, 0, y0.length);
            setStepStart(acceptStep(createInterpolator(forward, yDotK, getStepStart(), stateTmp, equations.getMapper()),
                                    finalTime));

            if (!isLastStep()) {

                // stepsize control for next step
                final T factor = MathUtils.min(maxGrowth,
                                               MathUtils.max(minReduction, safety.multiply(error.pow(exp))));
                final T  scaledH    = getStepSize().multiply(factor);
                final T  nextT      = getStepStart().getTime().add(scaledH);
                final boolean nextIsLast = forward ?
                                           nextT.subtract(finalTime).getReal() >= 0 :
                                           nextT.subtract(finalTime).getReal() <= 0;
                hNew = filterStep(scaledH, forward, nextIsLast);

                final T  filteredNextT      = getStepStart().getTime().add(hNew);
                final boolean filteredNextIsLast = forward ?
                                                   filteredNextT.subtract(finalTime).getReal() >= 0 :
                                                   filteredNextT.subtract(finalTime).getReal() <= 0;
                if (filteredNextIsLast) {
                    hNew = finalTime.subtract(getStepStart().getTime());
                }

            }

        } while (!isLastStep());

        final FieldODEStateAndDerivative<T> finalState = getStepStart();
        resetInternalState();
        return finalState;

    }

    
    public T getMinReduction() {
        return minReduction;
    }

    
    public void setMinReduction(final T minReduction) {
        this.minReduction = minReduction;
    }

    
    public T getMaxGrowth() {
        return maxGrowth;
    }

    
    public void setMaxGrowth(final T maxGrowth) {
        this.maxGrowth = maxGrowth;
    }

    
    protected abstract T estimateError(T[][] yDotK, T[] y0, T[] y1, T h);

}
