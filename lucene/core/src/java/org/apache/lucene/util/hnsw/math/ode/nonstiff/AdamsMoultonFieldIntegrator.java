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

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowFieldMatrix;
import org.apache.lucene.util.hnsw.math.linear.FieldMatrixPreservingVisitor;
import org.apache.lucene.util.hnsw.math.ode.FieldExpandableODE;
import org.apache.lucene.util.hnsw.math.ode.FieldODEState;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;



public class AdamsMoultonFieldIntegrator<T extends RealFieldElement<T>> extends AdamsFieldIntegrator<T> {

    
    private static final String METHOD_NAME = "Adams-Moulton";

    
    public AdamsMoultonFieldIntegrator(final Field<T> field, final int nSteps,
                                       final double minStep, final double maxStep,
                                       final double scalAbsoluteTolerance,
                                       final double scalRelativeTolerance)
        throws NumberIsTooSmallException {
        super(field, METHOD_NAME, nSteps, nSteps + 1, minStep, maxStep,
              scalAbsoluteTolerance, scalRelativeTolerance);
    }

    
    public AdamsMoultonFieldIntegrator(final Field<T> field, final int nSteps,
                                       final double minStep, final double maxStep,
                                       final double[] vecAbsoluteTolerance,
                                       final double[] vecRelativeTolerance)
        throws IllegalArgumentException {
        super(field, METHOD_NAME, nSteps, nSteps + 1, minStep, maxStep,
              vecAbsoluteTolerance, vecRelativeTolerance);
    }

    
    @Override
    public FieldODEStateAndDerivative<T> integrate(final FieldExpandableODE<T> equations,
                                                   final FieldODEState<T> initialState,
                                                   final T finalTime)
        throws NumberIsTooSmallException, DimensionMismatchException,
               MaxCountExceededException, NoBracketingException {

        sanityChecks(initialState, finalTime);
        final T   t0 = initialState.getTime();
        final T[] y  = equations.getMapper().mapState(initialState);
        setStepStart(initIntegration(equations, t0, y, finalTime));
        final boolean forward = finalTime.subtract(initialState.getTime()).getReal() > 0;

        // compute the initial Nordsieck vector using the configured starter integrator
        start(equations, getStepStart(), finalTime);

        // reuse the step that was chosen by the starter integrator
        FieldODEStateAndDerivative<T> stepStart = getStepStart();
        FieldODEStateAndDerivative<T> stepEnd   =
                        AdamsFieldStepInterpolator.taylor(stepStart,
                                                          stepStart.getTime().add(getStepSize()),
                                                          getStepSize(), scaled, nordsieck);

        // main integration loop
        setIsLastStep(false);
        do {

            T[] predictedY = null;
            final T[] predictedScaled = MathArrays.buildArray(getField(), y.length);
            Array2DRowFieldMatrix<T> predictedNordsieck = null;
            T error = getField().getZero().add(10);
            while (error.subtract(1.0).getReal() >= 0.0) {

                // predict a first estimate of the state at step end (P in the PECE sequence)
                predictedY = stepEnd.getState();

                // evaluate a first estimate of the derivative (first E in the PECE sequence)
                final T[] yDot = computeDerivatives(stepEnd.getTime(), predictedY);

                // update Nordsieck vector
                for (int j = 0; j < predictedScaled.length; ++j) {
                    predictedScaled[j] = getStepSize().multiply(yDot[j]);
                }
                predictedNordsieck = updateHighOrderDerivativesPhase1(nordsieck);
                updateHighOrderDerivativesPhase2(scaled, predictedScaled, predictedNordsieck);

                // apply correction (C in the PECE sequence)
                error = predictedNordsieck.walkInOptimizedOrder(new Corrector(y, predictedScaled, predictedY));

                if (error.subtract(1.0).getReal() >= 0.0) {
                    // reject the step and attempt to reduce error by stepsize control
                    final T factor = computeStepGrowShrinkFactor(error);
                    rescale(filterStep(getStepSize().multiply(factor), forward, false));
                    stepEnd = AdamsFieldStepInterpolator.taylor(getStepStart(),
                                                                getStepStart().getTime().add(getStepSize()),
                                                                getStepSize(),
                                                                scaled,
                                                                nordsieck);
                }
            }

            // evaluate a final estimate of the derivative (second E in the PECE sequence)
            final T[] correctedYDot = computeDerivatives(stepEnd.getTime(), predictedY);

            // update Nordsieck vector
            final T[] correctedScaled = MathArrays.buildArray(getField(), y.length);
            for (int j = 0; j < correctedScaled.length; ++j) {
                correctedScaled[j] = getStepSize().multiply(correctedYDot[j]);
            }
            updateHighOrderDerivativesPhase2(predictedScaled, correctedScaled, predictedNordsieck);

            // discrete events handling
            stepEnd = new FieldODEStateAndDerivative<T>(stepEnd.getTime(), predictedY, correctedYDot);
            setStepStart(acceptStep(new AdamsFieldStepInterpolator<T>(getStepSize(), stepEnd,
                                                                      correctedScaled, predictedNordsieck, forward,
                                                                      getStepStart(), stepEnd,
                                                                      equations.getMapper()),
                                    finalTime));
            scaled    = correctedScaled;
            nordsieck = predictedNordsieck;

            if (!isLastStep()) {

                System.arraycopy(predictedY, 0, y, 0, y.length);

                if (resetOccurred()) {
                    // some events handler has triggered changes that
                    // invalidate the derivatives, we need to restart from scratch
                    start(equations, getStepStart(), finalTime);
                }

                // stepsize control for next step
                final T  factor     = computeStepGrowShrinkFactor(error);
                final T  scaledH    = getStepSize().multiply(factor);
                final T  nextT      = getStepStart().getTime().add(scaledH);
                final boolean nextIsLast = forward ?
                                           nextT.subtract(finalTime).getReal() >= 0 :
                                           nextT.subtract(finalTime).getReal() <= 0;
                T hNew = filterStep(scaledH, forward, nextIsLast);

                final T  filteredNextT      = getStepStart().getTime().add(hNew);
                final boolean filteredNextIsLast = forward ?
                                                   filteredNextT.subtract(finalTime).getReal() >= 0 :
                                                   filteredNextT.subtract(finalTime).getReal() <= 0;
                if (filteredNextIsLast) {
                    hNew = finalTime.subtract(getStepStart().getTime());
                }

                rescale(hNew);
                stepEnd = AdamsFieldStepInterpolator.taylor(getStepStart(), getStepStart().getTime().add(getStepSize()),
                                                            getStepSize(), scaled, nordsieck);

            }

        } while (!isLastStep());

        final FieldODEStateAndDerivative<T> finalState = getStepStart();
        setStepStart(null);
        setStepSize(null);
        return finalState;

    }

    
    private class Corrector implements FieldMatrixPreservingVisitor<T> {

        
        private final T[] previous;

        
        private final T[] scaled;

        
        private final T[] before;

        
        private final T[] after;

        
        Corrector(final T[] previous, final T[] scaled, final T[] state) {
            this.previous = previous;
            this.scaled   = scaled;
            this.after    = state;
            this.before   = state.clone();
        }

        
        public void start(int rows, int columns,
                          int startRow, int endRow, int startColumn, int endColumn) {
            Arrays.fill(after, getField().getZero());
        }

        
        public void visit(int row, int column, T value) {
            if ((row & 0x1) == 0) {
                after[column] = after[column].subtract(value);
            } else {
                after[column] = after[column].add(value);
            }
        }

        
        public T end() {

            T error = getField().getZero();
            for (int i = 0; i < after.length; ++i) {
                after[i] = after[i].add(previous[i].add(scaled[i]));
                if (i < mainSetDimension) {
                    final T yScale = MathUtils.max(previous[i].abs(), after[i].abs());
                    final T tol = (vecAbsoluteTolerance == null) ?
                                  yScale.multiply(scalRelativeTolerance).add(scalAbsoluteTolerance) :
                                  yScale.multiply(vecRelativeTolerance[i]).add(vecAbsoluteTolerance[i]);
                    final T ratio  = after[i].subtract(before[i]).divide(tol); // (corrected-predicted)/tol
                    error = error.add(ratio.multiply(ratio));
                }
            }

            return error.divide(mainSetDimension).sqrt();

        }
    }

}
