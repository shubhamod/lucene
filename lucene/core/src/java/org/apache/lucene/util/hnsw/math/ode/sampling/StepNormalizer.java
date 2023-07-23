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

import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;



public class StepNormalizer implements StepHandler {
    
    private double h;

    
    private final FixedStepHandler handler;

    
    private double firstTime;

    
    private double lastTime;

    
    private double[] lastState;

    
    private double[] lastDerivatives;

    
    private boolean forward;

    
    private final StepNormalizerBounds bounds;

    
    private final StepNormalizerMode mode;

    
    public StepNormalizer(final double h, final FixedStepHandler handler) {
        this(h, handler, StepNormalizerMode.INCREMENT,
             StepNormalizerBounds.FIRST);
    }

    
    public StepNormalizer(final double h, final FixedStepHandler handler,
                          final StepNormalizerMode mode) {
        this(h, handler, mode, StepNormalizerBounds.FIRST);
    }

    
    public StepNormalizer(final double h, final FixedStepHandler handler,
                          final StepNormalizerBounds bounds) {
        this(h, handler, StepNormalizerMode.INCREMENT, bounds);
    }

    
    public StepNormalizer(final double h, final FixedStepHandler handler,
                          final StepNormalizerMode mode,
                          final StepNormalizerBounds bounds) {
        this.h          = FastMath.abs(h);
        this.handler    = handler;
        this.mode       = mode;
        this.bounds     = bounds;
        firstTime       = Double.NaN;
        lastTime        = Double.NaN;
        lastState       = null;
        lastDerivatives = null;
        forward         = true;
    }

    
    public void init(double t0, double[] y0, double t) {

        firstTime       = Double.NaN;
        lastTime        = Double.NaN;
        lastState       = null;
        lastDerivatives = null;
        forward         = true;

        // initialize the underlying handler
        handler.init(t0, y0, t);

    }

    
    public void handleStep(final StepInterpolator interpolator, final boolean isLast)
        throws MaxCountExceededException {
        // The first time, update the last state with the start information.
        if (lastState == null) {
            firstTime = interpolator.getPreviousTime();
            lastTime = interpolator.getPreviousTime();
            interpolator.setInterpolatedTime(lastTime);
            lastState = interpolator.getInterpolatedState().clone();
            lastDerivatives = interpolator.getInterpolatedDerivatives().clone();

            // Take the integration direction into account.
            forward = interpolator.getCurrentTime() >= lastTime;
            if (!forward) {
                h = -h;
            }
        }

        // Calculate next normalized step time.
        double nextTime = (mode == StepNormalizerMode.INCREMENT) ?
                          lastTime + h :
                          (FastMath.floor(lastTime / h) + 1) * h;
        if (mode == StepNormalizerMode.MULTIPLES &&
            Precision.equals(nextTime, lastTime, 1)) {
            nextTime += h;
        }

        // Process normalized steps as long as they are in the current step.
        boolean nextInStep = isNextInStep(nextTime, interpolator);
        while (nextInStep) {
            // Output the stored previous step.
            doNormalizedStep(false);

            // Store the next step as last step.
            storeStep(interpolator, nextTime);

            // Move on to the next step.
            nextTime += h;
            nextInStep = isNextInStep(nextTime, interpolator);
        }

        if (isLast) {
            // There will be no more steps. The stored one should be given to
            // the handler. We may have to output one more step. Only the last
            // one of those should be flagged as being the last.
            boolean addLast = bounds.lastIncluded() &&
                              lastTime != interpolator.getCurrentTime();
            doNormalizedStep(!addLast);
            if (addLast) {
                storeStep(interpolator, interpolator.getCurrentTime());
                doNormalizedStep(true);
            }
        }
    }

    
    private boolean isNextInStep(double nextTime,
                                 StepInterpolator interpolator) {
        return forward ?
               nextTime <= interpolator.getCurrentTime() :
               nextTime >= interpolator.getCurrentTime();
    }

    
    private void doNormalizedStep(boolean isLast) {
        if (!bounds.firstIncluded() && firstTime == lastTime) {
            return;
        }
        handler.handleStep(lastTime, lastState, lastDerivatives, isLast);
    }

    
    private void storeStep(StepInterpolator interpolator, double t)
        throws MaxCountExceededException {
        lastTime = t;
        interpolator.setInterpolatedTime(lastTime);
        System.arraycopy(interpolator.getInterpolatedState(), 0,
                         lastState, 0, lastState.length);
        System.arraycopy(interpolator.getInterpolatedDerivatives(), 0,
                         lastDerivatives, 0, lastDerivatives.length);
    }
}
