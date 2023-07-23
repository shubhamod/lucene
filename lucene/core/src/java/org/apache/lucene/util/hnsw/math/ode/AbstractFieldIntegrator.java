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
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.analysis.solvers.BracketedRealFieldUnivariateSolver;
import org.apache.lucene.util.hnsw.math.analysis.solvers.FieldBracketingNthOrderBrentSolver;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.ode.events.FieldEventHandler;
import org.apache.lucene.util.hnsw.math.ode.events.FieldEventState;
import org.apache.lucene.util.hnsw.math.ode.sampling.AbstractFieldStepInterpolator;
import org.apache.lucene.util.hnsw.math.ode.sampling.FieldStepHandler;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.IntegerSequence;


public abstract class AbstractFieldIntegrator<T extends RealFieldElement<T>> implements FirstOrderFieldIntegrator<T> {

    
    private static final double DEFAULT_RELATIVE_ACCURACY = 1e-14;

    
    private static final double DEFAULT_FUNCTION_VALUE_ACCURACY = 1e-15;

    
    private Collection<FieldStepHandler<T>> stepHandlers;

    
    private FieldODEStateAndDerivative<T> stepStart;

    
    private T stepSize;

    
    private boolean isLastStep;

    
    private boolean resetOccurred;

    
    private final Field<T> field;

    
    private Collection<FieldEventState<T>> eventsStates;

    
    private boolean statesInitialized;

    
    private final String name;

    
    private IntegerSequence.Incrementor evaluations;

    
    private transient FieldExpandableODE<T> equations;

    
    protected AbstractFieldIntegrator(final Field<T> field, final String name) {
        this.field        = field;
        this.name         = name;
        stepHandlers      = new ArrayList<FieldStepHandler<T>>();
        stepStart         = null;
        stepSize          = null;
        eventsStates      = new ArrayList<FieldEventState<T>>();
        statesInitialized = false;
        evaluations       = IntegerSequence.Incrementor.create().withMaximalCount(Integer.MAX_VALUE);
    }

    
    public Field<T> getField() {
        return field;
    }

    
    public String getName() {
        return name;
    }

    
    public void addStepHandler(final FieldStepHandler<T> handler) {
        stepHandlers.add(handler);
    }

    
    public Collection<FieldStepHandler<T>> getStepHandlers() {
        return Collections.unmodifiableCollection(stepHandlers);
    }

    
    public void clearStepHandlers() {
        stepHandlers.clear();
    }

    
    public void addEventHandler(final FieldEventHandler<T> handler,
                                final double maxCheckInterval,
                                final double convergence,
                                final int maxIterationCount) {
        addEventHandler(handler, maxCheckInterval, convergence,
                        maxIterationCount,
                        new FieldBracketingNthOrderBrentSolver<T>(field.getZero().add(DEFAULT_RELATIVE_ACCURACY),
                                                                  field.getZero().add(convergence),
                                                                  field.getZero().add(DEFAULT_FUNCTION_VALUE_ACCURACY),
                                                                  5));
    }

    
    public void addEventHandler(final FieldEventHandler<T> handler,
                                final double maxCheckInterval,
                                final double convergence,
                                final int maxIterationCount,
                                final BracketedRealFieldUnivariateSolver<T> solver) {
        eventsStates.add(new FieldEventState<T>(handler, maxCheckInterval, field.getZero().add(convergence),
                                                maxIterationCount, solver));
    }

    
    public Collection<FieldEventHandler<T>> getEventHandlers() {
        final List<FieldEventHandler<T>> list = new ArrayList<FieldEventHandler<T>>(eventsStates.size());
        for (FieldEventState<T> state : eventsStates) {
            list.add(state.getEventHandler());
        }
        return Collections.unmodifiableCollection(list);
    }

    
    public void clearEventHandlers() {
        eventsStates.clear();
    }

    
    public FieldODEStateAndDerivative<T> getCurrentStepStart() {
        return stepStart;
    }

    
    public T getCurrentSignedStepsize() {
        return stepSize;
    }

    
    public void setMaxEvaluations(int maxEvaluations) {
        evaluations = evaluations.withMaximalCount((maxEvaluations < 0) ? Integer.MAX_VALUE : maxEvaluations);
    }

    
    public int getMaxEvaluations() {
        return evaluations.getMaximalCount();
    }

    
    public int getEvaluations() {
        return evaluations.getCount();
    }

    
    protected FieldODEStateAndDerivative<T> initIntegration(final FieldExpandableODE<T> eqn,
                                                            final T t0, final T[] y0, final T t) {

        this.equations = eqn;
        evaluations    = evaluations.withStart(0);

        // initialize ODE
        eqn.init(t0, y0, t);

        // set up derivatives of initial state
        final T[] y0Dot = computeDerivatives(t0, y0);
        final FieldODEStateAndDerivative<T> state0 = new FieldODEStateAndDerivative<T>(t0, y0, y0Dot);

        // initialize event handlers
        for (final FieldEventState<T> state : eventsStates) {
            state.getEventHandler().init(state0, t);
        }

        // initialize step handlers
        for (FieldStepHandler<T> handler : stepHandlers) {
            handler.init(state0, t);
        }

        setStateInitialized(false);

        return state0;

    }

    
    protected FieldExpandableODE<T> getEquations() {
        return equations;
    }

    
    protected IntegerSequence.Incrementor getEvaluationsCounter() {
        return evaluations;
    }

    
    public T[] computeDerivatives(final T t, final T[] y)
        throws DimensionMismatchException, MaxCountExceededException, NullPointerException {
        evaluations.increment();
        return equations.computeDerivatives(t, y);
    }

    
    protected void setStateInitialized(final boolean stateInitialized) {
        this.statesInitialized = stateInitialized;
    }

    
    protected FieldODEStateAndDerivative<T> acceptStep(final AbstractFieldStepInterpolator<T> interpolator,
                                                       final T tEnd)
        throws MaxCountExceededException, DimensionMismatchException, NoBracketingException {

            FieldODEStateAndDerivative<T> previousState = interpolator.getGlobalPreviousState();
            final FieldODEStateAndDerivative<T> currentState = interpolator.getGlobalCurrentState();

            // initialize the events states if needed
            if (! statesInitialized) {
                for (FieldEventState<T> state : eventsStates) {
                    state.reinitializeBegin(interpolator);
                }
                statesInitialized = true;
            }

            // search for next events that may occur during the step
            final int orderingSign = interpolator.isForward() ? +1 : -1;
            SortedSet<FieldEventState<T>> occurringEvents = new TreeSet<FieldEventState<T>>(new Comparator<FieldEventState<T>>() {

                
                public int compare(FieldEventState<T> es0, FieldEventState<T> es1) {
                    return orderingSign * Double.compare(es0.getEventTime().getReal(), es1.getEventTime().getReal());
                }

            });

            for (final FieldEventState<T> state : eventsStates) {
                if (state.evaluateStep(interpolator)) {
                    // the event occurs during the current step
                    occurringEvents.add(state);
                }
            }

            AbstractFieldStepInterpolator<T> restricted = interpolator;
            while (!occurringEvents.isEmpty()) {

                // handle the chronologically first event
                final Iterator<FieldEventState<T>> iterator = occurringEvents.iterator();
                final FieldEventState<T> currentEvent = iterator.next();
                iterator.remove();

                // get state at event time
                final FieldODEStateAndDerivative<T> eventState = restricted.getInterpolatedState(currentEvent.getEventTime());

                // restrict the interpolator to the first part of the step, up to the event
                restricted = restricted.restrictStep(previousState, eventState);

                // advance all event states to current time
                for (final FieldEventState<T> state : eventsStates) {
                    state.stepAccepted(eventState);
                    isLastStep = isLastStep || state.stop();
                }

                // handle the first part of the step, up to the event
                for (final FieldStepHandler<T> handler : stepHandlers) {
                    handler.handleStep(restricted, isLastStep);
                }

                if (isLastStep) {
                    // the event asked to stop integration
                    return eventState;
                }

                FieldODEState<T> newState = null;
                resetOccurred = false;
                for (final FieldEventState<T> state : eventsStates) {
                    newState = state.reset(eventState);
                    if (newState != null) {
                        // some event handler has triggered changes that
                        // invalidate the derivatives, we need to recompute them
                        final T[] y    = equations.getMapper().mapState(newState);
                        final T[] yDot = computeDerivatives(newState.getTime(), y);
                        resetOccurred = true;
                        return equations.getMapper().mapStateAndDerivative(newState.getTime(), y, yDot);
                    }
                }

                // prepare handling of the remaining part of the step
                previousState = eventState;
                restricted = restricted.restrictStep(eventState, currentState);

                // check if the same event occurs again in the remaining part of the step
                if (currentEvent.evaluateStep(restricted)) {
                    // the event occurs during the current step
                    occurringEvents.add(currentEvent);
                }

            }

            // last part of the step, after the last event
            for (final FieldEventState<T> state : eventsStates) {
                state.stepAccepted(currentState);
                isLastStep = isLastStep || state.stop();
            }
            isLastStep = isLastStep || currentState.getTime().subtract(tEnd).abs().getReal() <= FastMath.ulp(tEnd.getReal());

            // handle the remaining part of the step, after all events if any
            for (FieldStepHandler<T> handler : stepHandlers) {
                handler.handleStep(restricted, isLastStep);
            }

            return currentState;

    }

    
    protected void sanityChecks(final FieldODEState<T> eqn, final T t)
        throws NumberIsTooSmallException, DimensionMismatchException {

        final double threshold = 1000 * FastMath.ulp(FastMath.max(FastMath.abs(eqn.getTime().getReal()),
                                                                  FastMath.abs(t.getReal())));
        final double dt = eqn.getTime().subtract(t).abs().getReal();
        if (dt <= threshold) {
            throw new NumberIsTooSmallException(LocalizedFormats.TOO_SMALL_INTEGRATION_INTERVAL,
                                                dt, threshold, false);
        }

    }

    
    protected boolean resetOccurred() {
        return resetOccurred;
    }

    
    protected void setStepSize(final T stepSize) {
        this.stepSize = stepSize;
    }

    
    protected T getStepSize() {
        return stepSize;
    }
    
    protected void setStepStart(final FieldODEStateAndDerivative<T> stepStart) {
        this.stepStart = stepStart;
    }

    
    protected FieldODEStateAndDerivative<T> getStepStart() {
        return stepStart;
    }

    
    protected void setIsLastStep(final boolean isLastStep) {
        this.isLastStep = isLastStep;
    }

    
    protected boolean isLastStep() {
        return isLastStep;
    }

}
