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
import org.apache.lucene.util.hnsw.math.ode.AbstractFieldIntegrator;
import org.apache.lucene.util.hnsw.math.ode.FieldEquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.FieldExpandableODE;
import org.apache.lucene.util.hnsw.math.ode.FirstOrderFieldDifferentialEquations;
import org.apache.lucene.util.hnsw.math.ode.FieldODEState;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;
import org.apache.lucene.util.hnsw.math.util.MathArrays;



public abstract class RungeKuttaFieldIntegrator<T extends RealFieldElement<T>>
    extends AbstractFieldIntegrator<T>
    implements FieldButcherArrayProvider<T> {

    
    private final T[] c;

    
    private final T[][] a;

    
    private final T[] b;

    
    private final T step;

    
    protected RungeKuttaFieldIntegrator(final Field<T> field, final String name, final T step) {
        super(field, name);
        this.c    = getC();
        this.a    = getA();
        this.b    = getB();
        this.step = step.abs();
    }

    
    protected T fraction(final int p, final int q) {
        return getField().getZero().add(p).divide(q);
    }

    
    protected abstract RungeKuttaFieldStepInterpolator<T> createInterpolator(boolean forward, T[][] yDotK,
                                                                             final FieldODEStateAndDerivative<T> globalPreviousState,
                                                                             final FieldODEStateAndDerivative<T> globalCurrentState,
                                                                             FieldEquationsMapper<T> mapper);

    
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
        if (forward) {
            if (getStepStart().getTime().add(step).subtract(finalTime).getReal() >= 0) {
                setStepSize(finalTime.subtract(getStepStart().getTime()));
            } else {
                setStepSize(step);
            }
        } else {
            if (getStepStart().getTime().subtract(step).subtract(finalTime).getReal() <= 0) {
                setStepSize(finalTime.subtract(getStepStart().getTime()));
            } else {
                setStepSize(step.negate());
            }
        }

        // main integration loop
        setIsLastStep(false);
        do {

            // first stage
            y        = equations.getMapper().mapState(getStepStart());
            yDotK[0] = equations.getMapper().mapDerivative(getStepStart());

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
                T sum = yDotK[0][j].multiply(b[0]);
                for (int l = 1; l < stages; ++l) {
                    sum = sum.add(yDotK[l][j].multiply(b[l]));
                }
                yTmp[j] = y[j].add(getStepSize().multiply(sum));
            }
            final T stepEnd   = getStepStart().getTime().add(getStepSize());
            final T[] yDotTmp = computeDerivatives(stepEnd, yTmp);
            final FieldODEStateAndDerivative<T> stateTmp = new FieldODEStateAndDerivative<T>(stepEnd, yTmp, yDotTmp);

            // discrete events handling
            System.arraycopy(yTmp, 0, y, 0, y0.length);
            setStepStart(acceptStep(createInterpolator(forward, yDotK, getStepStart(), stateTmp, equations.getMapper()),
                                    finalTime));

            if (!isLastStep()) {

                // stepsize control for next step
                final T  nextT      = getStepStart().getTime().add(getStepSize());
                final boolean nextIsLast = forward ?
                                           (nextT.subtract(finalTime).getReal() >= 0) :
                                           (nextT.subtract(finalTime).getReal() <= 0);
                if (nextIsLast) {
                    setStepSize(finalTime.subtract(getStepStart().getTime()));
                }
            }

        } while (!isLastStep());

        final FieldODEStateAndDerivative<T> finalState = getStepStart();
        setStepStart(null);
        setStepSize(null);
        return finalState;

    }

    
    public T[] singleStep(final FirstOrderFieldDifferentialEquations<T> equations,
                          final T t0, final T[] y0, final T t) {

        // create some internal working arrays
        final T[] y       = y0.clone();
        final int stages  = c.length + 1;
        final T[][] yDotK = MathArrays.buildArray(getField(), stages, -1);
        final T[] yTmp    = y0.clone();

        // first stage
        final T h = t.subtract(t0);
        yDotK[0] = equations.computeDerivatives(t0, y);

        // next stages
        for (int k = 1; k < stages; ++k) {

            for (int j = 0; j < y0.length; ++j) {
                T sum = yDotK[0][j].multiply(a[k-1][0]);
                for (int l = 1; l < k; ++l) {
                    sum = sum.add(yDotK[l][j].multiply(a[k-1][l]));
                }
                yTmp[j] = y[j].add(h.multiply(sum));
            }

            yDotK[k] = equations.computeDerivatives(t0.add(h.multiply(c[k-1])), yTmp);

        }

        // estimate the state at the end of the step
        for (int j = 0; j < y0.length; ++j) {
            T sum = yDotK[0][j].multiply(b[0]);
            for (int l = 1; l < stages; ++l) {
                sum = sum.add(yDotK[l][j].multiply(b[l]));
            }
            y[j] = y[j].add(h.multiply(sum));
        }

        return y;

    }

}
