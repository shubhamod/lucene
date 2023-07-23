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

package org.apache.lucene.util.hnsw.math.ode.sampling;

import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;



public class FieldStepNormalizer<T extends RealFieldElement<T>> implements FieldStepHandler<T> {

    
    private double h;

    
    private final FieldFixedStepHandler<T> handler;

    
    private FieldODEStateAndDerivative<T> first;

    
    private FieldODEStateAndDerivative<T> last;

    
    private boolean forward;

    
    private final StepNormalizerBounds bounds;

    
    private final StepNormalizerMode mode;

    
    public FieldStepNormalizer(final double h, final FieldFixedStepHandler<T> handler) {
        this(h, handler, StepNormalizerMode.INCREMENT,
             StepNormalizerBounds.FIRST);
    }

    
    public FieldStepNormalizer(final double h, final FieldFixedStepHandler<T> handler,
                               final StepNormalizerMode mode) {
        this(h, handler, mode, StepNormalizerBounds.FIRST);
    }

    
    public FieldStepNormalizer(final double h, final FieldFixedStepHandler<T> handler,
                               final StepNormalizerBounds bounds) {
        this(h, handler, StepNormalizerMode.INCREMENT, bounds);
    }

    
    public FieldStepNormalizer(final double h, final FieldFixedStepHandler<T> handler,
                               final StepNormalizerMode mode, final StepNormalizerBounds bounds) {
        this.h       = FastMath.abs(h);
        this.handler = handler;
        this.mode    = mode;
        this.bounds  = bounds;
        first        = null;
        last         = null;
        forward      = true;
    }

    
    public void init(final FieldODEStateAndDerivative<T> initialState, final T finalTime) {

        first   = null;
        last    = null;
        forward = true;

        // initialize the underlying handler
        handler.init(initialState, finalTime);

    }

    
    public void handleStep(final FieldStepInterpolator<T> interpolator, final boolean isLast)
        throws MaxCountExceededException {
        // The first time, update the last state with the start information.
        if (last == null) {

            first   = interpolator.getPreviousState();
            last    = first;

            // Take the integration direction into account.
            forward = interpolator.isForward();
            if (!forward) {
                h = -h;
            }
        }

        // Calculate next normalized step time.
        T nextTime = (mode == StepNormalizerMode.INCREMENT) ?
                     last.getTime().add(h) :
                     last.getTime().getField().getZero().add((FastMath.floor(last.getTime().getReal() / h) + 1) * h);
        if (mode == StepNormalizerMode.MULTIPLES &&
            Precision.equals(nextTime.getReal(), last.getTime().getReal(), 1)) {
            nextTime = nextTime.add(h);
        }

        // Process normalized steps as long as they are in the current step.
        boolean nextInStep = isNextInStep(nextTime, interpolator);
        while (nextInStep) {
            // Output the stored previous step.
            doNormalizedStep(false);

            // Store the next step as last step.
            last = interpolator.getInterpolatedState(nextTime);

            // Move on to the next step.
            nextTime = nextTime.add(h);
            nextInStep = isNextInStep(nextTime, interpolator);
        }

        if (isLast) {
            // There will be no more steps. The stored one should be given to
            // the handler. We may have to output one more step. Only the last
            // one of those should be flagged as being the last.
            final boolean addLast = bounds.lastIncluded() &&
                                    last.getTime().getReal() != interpolator.getCurrentState().getTime().getReal();
            doNormalizedStep(!addLast);
            if (addLast) {
                last = interpolator.getCurrentState();
                doNormalizedStep(true);
            }
        }
    }

    
    private boolean isNextInStep(final T nextTime, final FieldStepInterpolator<T> interpolator) {
        return forward ?
               nextTime.getReal() <= interpolator.getCurrentState().getTime().getReal() :
               nextTime.getReal() >= interpolator.getCurrentState().getTime().getReal();
    }

    
    private void doNormalizedStep(final boolean isLast) {
        if (!bounds.firstIncluded() && first.getTime().getReal() == last.getTime().getReal()) {
            return;
        }
        handler.handleStep(last, isLast);
    }

}
