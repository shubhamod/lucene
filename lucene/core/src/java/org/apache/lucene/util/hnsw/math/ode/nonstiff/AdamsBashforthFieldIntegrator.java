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
import org.apache.lucene.util.hnsw.math.linear.Array2DRowFieldMatrix;
import org.apache.lucene.util.hnsw.math.linear.FieldMatrix;
import org.apache.lucene.util.hnsw.math.ode.FieldExpandableODE;
import org.apache.lucene.util.hnsw.math.ode.FieldODEState;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;
import org.apache.lucene.util.hnsw.math.util.MathArrays;



public class AdamsBashforthFieldIntegrator<T extends RealFieldElement<T>> extends AdamsFieldIntegrator<T> {

    
    private static final String METHOD_NAME = "Adams-Bashforth";

    
    public AdamsBashforthFieldIntegrator(final Field<T> field, final int nSteps,
                                         final double minStep, final double maxStep,
                                         final double scalAbsoluteTolerance,
                                         final double scalRelativeTolerance)
        throws NumberIsTooSmallException {
        super(field, METHOD_NAME, nSteps, nSteps, minStep, maxStep,
              scalAbsoluteTolerance, scalRelativeTolerance);
    }

    
    public AdamsBashforthFieldIntegrator(final Field<T> field, final int nSteps,
                                         final double minStep, final double maxStep,
                                         final double[] vecAbsoluteTolerance,
                                         final double[] vecRelativeTolerance)
        throws IllegalArgumentException {
        super(field, METHOD_NAME, nSteps, nSteps, minStep, maxStep,
              vecAbsoluteTolerance, vecRelativeTolerance);
    }

    
    private T errorEstimation(final T[] previousState,
                              final T[] predictedState,
                              final T[] predictedScaled,
                              final FieldMatrix<T> predictedNordsieck) {

        T error = getField().getZero();
        for (int i = 0; i < mainSetDimension; ++i) {
            final T yScale = predictedState[i].abs();
            final T tol = (vecAbsoluteTolerance == null) ?
                          yScale.multiply(scalRelativeTolerance).add(scalAbsoluteTolerance) :
                          yScale.multiply(vecRelativeTolerance[i]).add(vecAbsoluteTolerance[i]);

            // apply Taylor formula from high order to low order,
            // for the sake of numerical accuracy
            T variation = getField().getZero();
            int sign = predictedNordsieck.getRowDimension() % 2 == 0 ? -1 : 1;
            for (int k = predictedNordsieck.getRowDimension() - 1; k >= 0; --k) {
                variation = variation.add(predictedNordsieck.getEntry(k, i).multiply(sign));
                sign      = -sign;
            }
            variation = variation.subtract(predictedScaled[i]);

            final T ratio  = predictedState[i].subtract(previousState[i]).add(variation).divide(tol);
            error = error.add(ratio.multiply(ratio));

        }

        return error.divide(mainSetDimension).sqrt();

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

                // predict a first estimate of the state at step end
                predictedY = stepEnd.getState();

                // evaluate the derivative
                final T[] yDot = computeDerivatives(stepEnd.getTime(), predictedY);

                // predict Nordsieck vector at step end
                for (int j = 0; j < predictedScaled.length; ++j) {
                    predictedScaled[j] = getStepSize().multiply(yDot[j]);
                }
                predictedNordsieck = updateHighOrderDerivativesPhase1(nordsieck);
                updateHighOrderDerivativesPhase2(scaled, predictedScaled, predictedNordsieck);

                // evaluate error
                error = errorEstimation(y, predictedY, predictedScaled, predictedNordsieck);

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

            // discrete events handling
            setStepStart(acceptStep(new AdamsFieldStepInterpolator<T>(getStepSize(), stepEnd,
                                                                      predictedScaled, predictedNordsieck, forward,
                                                                      getStepStart(), stepEnd,
                                                                      equations.getMapper()),
                                    finalTime));
            scaled    = predictedScaled;
            nordsieck = predictedNordsieck;

            if (!isLastStep()) {

                System.arraycopy(predictedY, 0, y, 0, y.length);

                if (resetOccurred()) {
                    // some events handler has triggered changes that
                    // invalidate the derivatives, we need to restart from scratch
                    start(equations, getStepStart(), finalTime);
                }

                // stepsize control for next step
                final T       factor     = computeStepGrowShrinkFactor(error);
                final T       scaledH    = getStepSize().multiply(factor);
                final T       nextT      = getStepStart().getTime().add(scaledH);
                final boolean nextIsLast = forward ?
                                           nextT.subtract(finalTime).getReal() >= 0 :
                                           nextT.subtract(finalTime).getReal() <= 0;
                T hNew = filterStep(scaledH, forward, nextIsLast);

                final T       filteredNextT      = getStepStart().getTime().add(hNew);
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

}
