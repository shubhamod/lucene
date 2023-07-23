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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealMatrixPreservingVisitor;
import org.apache.lucene.util.hnsw.math.ode.EquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.ExpandableStatefulODE;
import org.apache.lucene.util.hnsw.math.ode.sampling.NordsieckStepInterpolator;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class AdamsMoultonIntegrator extends AdamsIntegrator {

    
    private static final String METHOD_NAME = "Adams-Moulton";

    
    public AdamsMoultonIntegrator(final int nSteps,
                                  final double minStep, final double maxStep,
                                  final double scalAbsoluteTolerance,
                                  final double scalRelativeTolerance)
        throws NumberIsTooSmallException {
        super(METHOD_NAME, nSteps, nSteps + 1, minStep, maxStep,
              scalAbsoluteTolerance, scalRelativeTolerance);
    }

    
    public AdamsMoultonIntegrator(final int nSteps,
                                  final double minStep, final double maxStep,
                                  final double[] vecAbsoluteTolerance,
                                  final double[] vecRelativeTolerance)
        throws IllegalArgumentException {
        super(METHOD_NAME, nSteps, nSteps + 1, minStep, maxStep,
              vecAbsoluteTolerance, vecRelativeTolerance);
    }

    
    @Override
    public void integrate(final ExpandableStatefulODE equations,final double t)
        throws NumberIsTooSmallException, DimensionMismatchException,
               MaxCountExceededException, NoBracketingException {

        sanityChecks(equations, t);
        setEquations(equations);
        final boolean forward = t > equations.getTime();

        // initialize working arrays
        final double[] y0   = equations.getCompleteState();
        final double[] y    = y0.clone();
        final double[] yDot = new double[y.length];
        final double[] yTmp = new double[y.length];
        final double[] predictedScaled = new double[y.length];
        Array2DRowRealMatrix nordsieckTmp = null;

        // set up two interpolators sharing the integrator arrays
        final NordsieckStepInterpolator interpolator = new NordsieckStepInterpolator();
        interpolator.reinitialize(y, forward,
                                  equations.getPrimaryMapper(), equations.getSecondaryMappers());

        // set up integration control objects
        initIntegration(equations.getTime(), y0, t);

        // compute the initial Nordsieck vector using the configured starter integrator
        start(equations.getTime(), y, t);
        interpolator.reinitialize(stepStart, stepSize, scaled, nordsieck);
        interpolator.storeTime(stepStart);

        double hNew = stepSize;
        interpolator.rescale(hNew);

        isLastStep = false;
        do {

            double error = 10;
            while (error >= 1.0) {

                stepSize = hNew;

                // predict a first estimate of the state at step end (P in the PECE sequence)
                final double stepEnd = stepStart + stepSize;
                interpolator.setInterpolatedTime(stepEnd);
                final ExpandableStatefulODE expandable = getExpandable();
                final EquationsMapper primary = expandable.getPrimaryMapper();
                primary.insertEquationData(interpolator.getInterpolatedState(), yTmp);
                int index = 0;
                for (final EquationsMapper secondary : expandable.getSecondaryMappers()) {
                    secondary.insertEquationData(interpolator.getInterpolatedSecondaryState(index), yTmp);
                    ++index;
                }

                // evaluate a first estimate of the derivative (first E in the PECE sequence)
                computeDerivatives(stepEnd, yTmp, yDot);

                // update Nordsieck vector
                for (int j = 0; j < y0.length; ++j) {
                    predictedScaled[j] = stepSize * yDot[j];
                }
                nordsieckTmp = updateHighOrderDerivativesPhase1(nordsieck);
                updateHighOrderDerivativesPhase2(scaled, predictedScaled, nordsieckTmp);

                // apply correction (C in the PECE sequence)
                error = nordsieckTmp.walkInOptimizedOrder(new Corrector(y, predictedScaled, yTmp));

                if (error >= 1.0) {
                    // reject the step and attempt to reduce error by stepsize control
                    final double factor = computeStepGrowShrinkFactor(error);
                    hNew = filterStep(stepSize * factor, forward, false);
                    interpolator.rescale(hNew);
                }
            }

            // evaluate a final estimate of the derivative (second E in the PECE sequence)
            final double stepEnd = stepStart + stepSize;
            computeDerivatives(stepEnd, yTmp, yDot);

            // update Nordsieck vector
            final double[] correctedScaled = new double[y0.length];
            for (int j = 0; j < y0.length; ++j) {
                correctedScaled[j] = stepSize * yDot[j];
            }
            updateHighOrderDerivativesPhase2(predictedScaled, correctedScaled, nordsieckTmp);

            // discrete events handling
            System.arraycopy(yTmp, 0, y, 0, y.length);
            interpolator.reinitialize(stepEnd, stepSize, correctedScaled, nordsieckTmp);
            interpolator.storeTime(stepStart);
            interpolator.shift();
            interpolator.storeTime(stepEnd);
            stepStart = acceptStep(interpolator, y, yDot, t);
            scaled    = correctedScaled;
            nordsieck = nordsieckTmp;

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

    
    private class Corrector implements RealMatrixPreservingVisitor {

        
        private final double[] previous;

        
        private final double[] scaled;

        
        private final double[] before;

        
        private final double[] after;

        
        Corrector(final double[] previous, final double[] scaled, final double[] state) {
            this.previous = previous;
            this.scaled   = scaled;
            this.after    = state;
            this.before   = state.clone();
        }

        
        public void start(int rows, int columns,
                          int startRow, int endRow, int startColumn, int endColumn) {
            Arrays.fill(after, 0.0);
        }

        
        public void visit(int row, int column, double value) {
            if ((row & 0x1) == 0) {
                after[column] -= value;
            } else {
                after[column] += value;
            }
        }

        
        public double end() {

            double error = 0;
            for (int i = 0; i < after.length; ++i) {
                after[i] += previous[i] + scaled[i];
                if (i < mainSetDimension) {
                    final double yScale = FastMath.max(FastMath.abs(previous[i]), FastMath.abs(after[i]));
                    final double tol    = (vecAbsoluteTolerance == null) ?
                                          (scalAbsoluteTolerance + scalRelativeTolerance * yScale) :
                                          (vecAbsoluteTolerance[i] + vecRelativeTolerance[i] * yScale);
                    final double ratio  = (after[i] - before[i]) / tol; // (corrected-predicted)/tol
                    error += ratio * ratio;
                }
            }

            return FastMath.sqrt(error / mainSetDimension);

        }
    }

}
