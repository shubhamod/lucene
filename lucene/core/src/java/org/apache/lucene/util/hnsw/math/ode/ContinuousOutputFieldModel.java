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

package org.apache.lucene.util.hnsw.math.ode;

import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.ode.sampling.FieldStepHandler;
import org.apache.lucene.util.hnsw.math.ode.sampling.FieldStepInterpolator;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class ContinuousOutputFieldModel<T extends RealFieldElement<T>>
    implements FieldStepHandler<T> {

    
    private T initialTime;

    
    private T finalTime;

    
    private boolean forward;

    
    private int index;

    
    private List<FieldStepInterpolator<T>> steps;

    
    public ContinuousOutputFieldModel() {
        steps       = new ArrayList<FieldStepInterpolator<T>>();
        initialTime = null;
        finalTime   = null;
        forward     = true;
        index       = 0;
    }

    
    public void append(final ContinuousOutputFieldModel<T> model)
        throws MathIllegalArgumentException, MaxCountExceededException {

        if (model.steps.size() == 0) {
            return;
        }

        if (steps.size() == 0) {
            initialTime = model.initialTime;
            forward     = model.forward;
        } else {

            // safety checks
            final FieldODEStateAndDerivative<T> s1 = steps.get(0).getPreviousState();
            final FieldODEStateAndDerivative<T> s2 = model.steps.get(0).getPreviousState();
            checkDimensionsEquality(s1.getStateDimension(), s2.getStateDimension());
            checkDimensionsEquality(s1.getNumberOfSecondaryStates(), s2.getNumberOfSecondaryStates());
            for (int i = 0; i < s1.getNumberOfSecondaryStates(); ++i) {
                checkDimensionsEquality(s1.getSecondaryStateDimension(i), s2.getSecondaryStateDimension(i));
            }

            if (forward ^ model.forward) {
                throw new MathIllegalArgumentException(LocalizedFormats.PROPAGATION_DIRECTION_MISMATCH);
            }

            final FieldStepInterpolator<T> lastInterpolator = steps.get(index);
            final T current  = lastInterpolator.getCurrentState().getTime();
            final T previous = lastInterpolator.getPreviousState().getTime();
            final T step = current.subtract(previous);
            final T gap = model.getInitialTime().subtract(current);
            if (gap.abs().subtract(step.abs().multiply(1.0e-3)).getReal() > 0) {
                throw new MathIllegalArgumentException(LocalizedFormats.HOLE_BETWEEN_MODELS_TIME_RANGES,
                                                       gap.abs().getReal());
            }

        }

        for (FieldStepInterpolator<T> interpolator : model.steps) {
            steps.add(interpolator);
        }

        index = steps.size() - 1;
        finalTime = (steps.get(index)).getCurrentState().getTime();

    }

    
    private void checkDimensionsEquality(final int d1, final int d2)
        throws DimensionMismatchException {
        if (d1 != d2) {
            throw new DimensionMismatchException(d2, d1);
        }
    }

    
    public void init(final FieldODEStateAndDerivative<T> initialState, final T t) {
        initialTime = initialState.getTime();
        finalTime   = t;
        forward     = true;
        index       = 0;
        steps.clear();
    }

    
    public void handleStep(final FieldStepInterpolator<T> interpolator, final boolean isLast)
        throws MaxCountExceededException {

        if (steps.size() == 0) {
            initialTime = interpolator.getPreviousState().getTime();
            forward     = interpolator.isForward();
        }

        steps.add(interpolator);

        if (isLast) {
            finalTime = interpolator.getCurrentState().getTime();
            index     = steps.size() - 1;
        }

    }

    
    public T getInitialTime() {
        return initialTime;
    }

    
    public T getFinalTime() {
        return finalTime;
    }

    
    public FieldODEStateAndDerivative<T> getInterpolatedState(final T time) {

        // initialize the search with the complete steps table
        int iMin = 0;
        final FieldStepInterpolator<T> sMin = steps.get(iMin);
        T tMin = sMin.getPreviousState().getTime().add(sMin.getCurrentState().getTime()).multiply(0.5);

        int iMax = steps.size() - 1;
        final FieldStepInterpolator<T> sMax = steps.get(iMax);
        T tMax = sMax.getPreviousState().getTime().add(sMax.getCurrentState().getTime()).multiply(0.5);

        // handle points outside of the integration interval
        // or in the first and last step
        if (locatePoint(time, sMin) <= 0) {
            index = iMin;
            return sMin.getInterpolatedState(time);
        }
        if (locatePoint(time, sMax) >= 0) {
            index = iMax;
            return sMax.getInterpolatedState(time);
        }

        // reduction of the table slice size
        while (iMax - iMin > 5) {

            // use the last estimated index as the splitting index
            final FieldStepInterpolator<T> si = steps.get(index);
            final int location = locatePoint(time, si);
            if (location < 0) {
                iMax = index;
                tMax = si.getPreviousState().getTime().add(si.getCurrentState().getTime()).multiply(0.5);
            } else if (location > 0) {
                iMin = index;
                tMin = si.getPreviousState().getTime().add(si.getCurrentState().getTime()).multiply(0.5);
            } else {
                // we have found the target step, no need to continue searching
                return si.getInterpolatedState(time);
            }

            // compute a new estimate of the index in the reduced table slice
            final int iMed = (iMin + iMax) / 2;
            final FieldStepInterpolator<T> sMed = steps.get(iMed);
            final T tMed = sMed.getPreviousState().getTime().add(sMed.getCurrentState().getTime()).multiply(0.5);

            if (tMed.subtract(tMin).abs().subtract(1.0e-6).getReal() < 0 ||
                tMax.subtract(tMed).abs().subtract(1.0e-6).getReal() < 0) {
                // too close to the bounds, we estimate using a simple dichotomy
                index = iMed;
            } else {
                // estimate the index using a reverse quadratic polynomial
                // (reverse means we have i = P(t), thus allowing to simply
                // compute index = P(time) rather than solving a quadratic equation)
                final T d12 = tMax.subtract(tMed);
                final T d23 = tMed.subtract(tMin);
                final T d13 = tMax.subtract(tMin);
                final T dt1 = time.subtract(tMax);
                final T dt2 = time.subtract(tMed);
                final T dt3 = time.subtract(tMin);
                final T iLagrange =           dt2.multiply(dt3).multiply(d23).multiply(iMax).
                                     subtract(dt1.multiply(dt3).multiply(d13).multiply(iMed)).
                                     add(     dt1.multiply(dt2).multiply(d12).multiply(iMin)).
                                     divide(d12.multiply(d23).multiply(d13));
                index = (int) FastMath.rint(iLagrange.getReal());
            }

            // force the next size reduction to be at least one tenth
            final int low  = FastMath.max(iMin + 1, (9 * iMin + iMax) / 10);
            final int high = FastMath.min(iMax - 1, (iMin + 9 * iMax) / 10);
            if (index < low) {
                index = low;
            } else if (index > high) {
                index = high;
            }

        }

        // now the table slice is very small, we perform an iterative search
        index = iMin;
        while (index <= iMax && locatePoint(time, steps.get(index)) > 0) {
            ++index;
        }

        return steps.get(index).getInterpolatedState(time);

    }

    
    private int locatePoint(final T time, final FieldStepInterpolator<T> interval) {
        if (forward) {
            if (time.subtract(interval.getPreviousState().getTime()).getReal() < 0) {
                return -1;
            } else if (time.subtract(interval.getCurrentState().getTime()).getReal() > 0) {
                return +1;
            } else {
                return 0;
            }
        }
        if (time.subtract(interval.getPreviousState().getTime()).getReal() > 0) {
            return -1;
        } else if (time.subtract(interval.getCurrentState().getTime()).getReal() < 0) {
            return +1;
        } else {
            return 0;
        }
    }

}
