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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.ode.EquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.ExpandableStatefulODE;
import org.apache.lucene.util.hnsw.math.ode.sampling.NordsieckStepInterpolator;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class AdamsBashforthIntegrator extends AdamsIntegrator {

    
    private static final String METHOD_NAME = "Adams-Bashforth";

    
    public AdamsBashforthIntegrator(final int nSteps,
                                    final double minStep, final double maxStep,
                                    final double scalAbsoluteTolerance,
                                    final double scalRelativeTolerance)
        throws NumberIsTooSmallException {
        super(METHOD_NAME, nSteps, nSteps, minStep, maxStep,
              scalAbsoluteTolerance, scalRelativeTolerance);
    }

    
    public AdamsBashforthIntegrator(final int nSteps,
                                    final double minStep, final double maxStep,
                                    final double[] vecAbsoluteTolerance,
                                    final double[] vecRelativeTolerance)
        throws IllegalArgumentException {
        super(METHOD_NAME, nSteps, nSteps, minStep, maxStep,
              vecAbsoluteTolerance, vecRelativeTolerance);
    }

    
    private double errorEstimation(final double[] previousState,
                                   final double[] predictedState,
                                   final double[] predictedScaled,
                                   final RealMatrix predictedNordsieck) {

        double error = 0;
        for (int i = 0; i < mainSetDimension; ++i) {
            final double yScale = FastMath.abs(predictedState[i]);
            final double tol = (vecAbsoluteTolerance == null) ?
                               (scalAbsoluteTolerance + scalRelativeTolerance * yScale) :
                               (vecAbsoluteTolerance[i] + vecRelativeTolerance[i] * yScale);

            // apply Taylor formula from high order to low order,
            // for the sake of numerical accuracy
            double variation = 0;
            int sign = predictedNordsieck.getRowDimension() % 2 == 0 ? -1 : 1;
            for (int k = predictedNordsieck.getRowDimension() - 1; k >= 0; --k) {
                variation += sign * predictedNordsieck.getEntry(k, i);
                sign       = -sign;
            }
            variation -= predictedScaled[i];

            final double ratio  = (predictedState[i] - previousState[i] + variation) / tol;
            error              += ratio * ratio;

        }

        return FastMath.sqrt(error / mainSetDimension);

    }

    
    @Override
    public void integrate(final ExpandableStatefulODE equations, final double t)
        throws NumberIsTooSmallException, DimensionMismatchException,
               MaxCountExceededException, NoBracketingException {

        sanityChecks(equations, t);
        setEquations(equations);
        final boolean forward = t > equations.getTime();

        // initialize working arrays
        final double[] y    = equations.getCompleteState();
        final double[] yDot = new double[y.length];

        // set up an interpolator sharing the integrator arrays
        final NordsieckStepInterpolator interpolator = new NordsieckStepInterpolator();
        interpolator.reinitialize(y, forward,
                                  equations.getPrimaryMapper(), equations.getSecondaryMappers());

        // set up integration control objects
        initIntegration(equations.getTime(), y, t);

        // compute the initial Nordsieck vector using the configured starter integrator
        start(equations.getTime(), y, t);
        interpolator.reinitialize(stepStart, stepSize, scaled, nordsieck);
        interpolator.storeTime(stepStart);

        // reuse the step that was chosen by the starter integrator
        double hNew = stepSize;
        interpolator.rescale(hNew);

        // main integration loop
        isLastStep = false;
        do {

            interpolator.shift();
            final double[] predictedY      = new double[y.length];
            final double[] predictedScaled = new double[y.length];
            Array2DRowRealMatrix predictedNordsieck = null;
            double error = 10;
            while (error >= 1.0) {

                // predict a first estimate of the state at step end
                final double stepEnd = stepStart + hNew;
                interpolator.storeTime(stepEnd);
                final ExpandableStatefulODE expandable = getExpandable();
                final EquationsMapper primary = expandable.getPrimaryMapper();
                primary.insertEquationData(interpolator.getInterpolatedState(), predictedY);
                int index = 0;
                for (final EquationsMapper secondary : expandable.getSecondaryMappers()) {
                    secondary.insertEquationData(interpolator.getInterpolatedSecondaryState(index), predictedY);
                    ++index;
                }

                // evaluate the derivative
                computeDerivatives(stepEnd, predictedY, yDot);

                // predict Nordsieck vector at step end
                for (int j = 0; j < predictedScaled.length; ++j) {
                    predictedScaled[j] = hNew * yDot[j];
                }
                predictedNordsieck = updateHighOrderDerivativesPhase1(nordsieck);
                updateHighOrderDerivativesPhase2(scaled, predictedScaled, predictedNordsieck);

                // evaluate error
                error = errorEstimation(y, predictedY, predictedScaled, predictedNordsieck);

                if (error >= 1.0) {
                    // reject the step and attempt to reduce error by stepsize control
                    final double factor = computeStepGrowShrinkFactor(error);
                    hNew = filterStep(hNew * factor, forward, false);
                    interpolator.rescale(hNew);

                }
            }

            stepSize = hNew;
            final double stepEnd = stepStart + stepSize;
            interpolator.reinitialize(stepEnd, stepSize, predictedScaled, predictedNordsieck);

            // discrete events handling
            interpolator.storeTime(stepEnd);
            System.arraycopy(predictedY, 0, y, 0, y.length);
            stepStart = acceptStep(interpolator, y, yDot, t);
            scaled    = predictedScaled;
            nordsieck = predictedNordsieck;
            interpolator.reinitialize(stepEnd, stepSize, scaled, nordsieck);

            if (!isLastStep) {

                // prepare next step
                interpolator.storeTime(stepStart);

                if (resetOccurred) {
                    // some events handler has triggered changes that
                    // invalidate the derivatives, we need to restart from scratch
                    start(stepStart, y, t);
                    interpolator.reinitialize(stepStart, stepSize, scaled, nordsieck);
                }

                // stepsize control for next step
                final double  factor     = computeStepGrowShrinkFactor(error);
                final double  scaledH    = stepSize * factor;
                final double  nextT      = stepStart + scaledH;
                final boolean nextIsLast = forward ? (nextT >= t) : (nextT <= t);
                hNew = filterStep(scaledH, forward, nextIsLast);

                final double  filteredNextT      = stepStart + hNew;
                final boolean filteredNextIsLast = forward ? (filteredNextT >= t) : (filteredNextT <= t);
                if (filteredNextIsLast) {
                    hNew = t - stepStart;
                }

                interpolator.rescale(hNew);

            }

        } while (!isLastStep);

        // dispatch results
        equations.setTime(stepStart);
        equations.setCompleteState(y);

        resetInternalState();

    }

}
