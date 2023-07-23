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

import org.apache.lucene.util.hnsw.math.analysis.solvers.BracketingNthOrderBrentSolver;
import org.apache.lucene.util.hnsw.math.analysis.solvers.UnivariateSolver;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.ode.events.EventHandler;
import org.apache.lucene.util.hnsw.math.ode.events.EventState;
import org.apache.lucene.util.hnsw.math.ode.sampling.AbstractStepInterpolator;
import org.apache.lucene.util.hnsw.math.ode.sampling.StepHandler;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.IntegerSequence;
import org.apache.lucene.util.hnsw.math.util.Precision;


public abstract class AbstractIntegrator implements FirstOrderIntegrator {

    
    protected Collection<StepHandler> stepHandlers;

    
    protected double stepStart;

    
    protected double stepSize;

    
    protected boolean isLastStep;

    
    protected boolean resetOccurred;

    
    private Collection<EventState> eventsStates;

    
    private boolean statesInitialized;

    
    private final String name;

    
    private IntegerSequence.Incrementor evaluations;

    
    private transient ExpandableStatefulODE expandable;

    
    public AbstractIntegrator(final String name) {
        this.name = name;
        stepHandlers = new ArrayList<StepHandler>();
        stepStart = Double.NaN;
        stepSize  = Double.NaN;
        eventsStates = new ArrayList<EventState>();
        statesInitialized = false;
        evaluations = IntegerSequence.Incrementor.create().withMaximalCount(Integer.MAX_VALUE);
    }

    
    protected AbstractIntegrator() {
        this(null);
    }

    
    public String getName() {
        return name;
    }

    
    public void addStepHandler(final StepHandler handler) {
        stepHandlers.add(handler);
    }

    
    public Collection<StepHandler> getStepHandlers() {
        return Collections.unmodifiableCollection(stepHandlers);
    }

    
    public void clearStepHandlers() {
        stepHandlers.clear();
    }

    
    public void addEventHandler(final EventHandler handler,
                                final double maxCheckInterval,
                                final double convergence,
                                final int maxIterationCount) {
        addEventHandler(handler, maxCheckInterval, convergence,
                        maxIterationCount,
                        new BracketingNthOrderBrentSolver(convergence, 5));
    }

    
    public void addEventHandler(final EventHandler handler,
                                final double maxCheckInterval,
                                final double convergence,
                                final int maxIterationCount,
                                final UnivariateSolver solver) {
        eventsStates.add(new EventState(handler, maxCheckInterval, convergence,
                                        maxIterationCount, solver));
    }

    
    public Collection<EventHandler> getEventHandlers() {
        final List<EventHandler> list = new ArrayList<EventHandler>(eventsStates.size());
        for (EventState state : eventsStates) {
            list.add(state.getEventHandler());
        }
        return Collections.unmodifiableCollection(list);
    }

    
    public void clearEventHandlers() {
        eventsStates.clear();
    }

    
    public double getCurrentStepStart() {
        return stepStart;
    }

    
    public double getCurrentSignedStepsize() {
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

    
    protected void initIntegration(final double t0, final double[] y0, final double t) {

        evaluations = evaluations.withStart(0);

        for (final EventState state : eventsStates) {
            state.setExpandable(expandable);
            state.getEventHandler().init(t0, y0, t);
        }

        for (StepHandler handler : stepHandlers) {
            handler.init(t0, y0, t);
        }

        setStateInitialized(false);

    }

    
    protected void setEquations(final ExpandableStatefulODE equations) {
        this.expandable = equations;
    }

    
    protected ExpandableStatefulODE getExpandable() {
        return expandable;
    }

    
    @Deprecated
    protected org.apache.lucene.util.hnsw.math.util.Incrementor getEvaluationsCounter() {
        return org.apache.lucene.util.hnsw.math.util.Incrementor.wrap(evaluations);
    }

    
    protected IntegerSequence.Incrementor getCounter() {
        return evaluations;
    }

    
    public double integrate(final FirstOrderDifferentialEquations equations,
                            final double t0, final double[] y0, final double t, final double[] y)
        throws DimensionMismatchException, NumberIsTooSmallException,
               MaxCountExceededException, NoBracketingException {

        if (y0.length != equations.getDimension()) {
            throw new DimensionMismatchException(y0.length, equations.getDimension());
        }
        if (y.length != equations.getDimension()) {
            throw new DimensionMismatchException(y.length, equations.getDimension());
        }

        // prepare expandable stateful equations
        final ExpandableStatefulODE expandableODE = new ExpandableStatefulODE(equations);
        expandableODE.setTime(t0);
        expandableODE.setPrimaryState(y0);

        // perform integration
        integrate(expandableODE, t);

        // extract results back from the stateful equations
        System.arraycopy(expandableODE.getPrimaryState(), 0, y, 0, y.length);
        return expandableODE.getTime();

    }

    
    public abstract void integrate(ExpandableStatefulODE equations, double t)
        throws NumberIsTooSmallException, DimensionMismatchException,
               MaxCountExceededException, NoBracketingException;

    
    public void computeDerivatives(final double t, final double[] y, final double[] yDot)
        throws MaxCountExceededException, DimensionMismatchException, NullPointerException {
        evaluations.increment();
        expandable.computeDerivatives(t, y, yDot);
    }

    
    protected void setStateInitialized(final boolean stateInitialized) {
        this.statesInitialized = stateInitialized;
    }

    
    protected double acceptStep(final AbstractStepInterpolator interpolator,
                                final double[] y, final double[] yDot, final double tEnd)
        throws MaxCountExceededException, DimensionMismatchException, NoBracketingException {

            double previousT = interpolator.getGlobalPreviousTime();
            final double currentT = interpolator.getGlobalCurrentTime();

            // initialize the events states if needed
            if (! statesInitialized) {
                for (EventState state : eventsStates) {
                    state.reinitializeBegin(interpolator);
                }
                statesInitialized = true;
            }

            // search for next events that may occur during the step
            final int orderingSign = interpolator.isForward() ? +1 : -1;
            SortedSet<EventState> occurringEvents = new TreeSet<EventState>(new Comparator<EventState>() {

                
                public int compare(EventState es0, EventState es1) {
                    return orderingSign * Double.compare(es0.getEventTime(), es1.getEventTime());
                }

            });

            for (final EventState state : eventsStates) {
                if (state.evaluateStep(interpolator)) {
                    // the event occurs during the current step
                    occurringEvents.add(state);
                }
            }

            while (!occurringEvents.isEmpty()) {

                // handle the chronologically first event
                final Iterator<EventState> iterator = occurringEvents.iterator();
                final EventState currentEvent = iterator.next();
                iterator.remove();

                // restrict the interpolator to the first part of the step, up to the event
                final double eventT = currentEvent.getEventTime();
                interpolator.setSoftPreviousTime(previousT);
                interpolator.setSoftCurrentTime(eventT);

                // get state at event time
                interpolator.setInterpolatedTime(eventT);
                final double[] eventYComplete = new double[y.length];
                expandable.getPrimaryMapper().insertEquationData(interpolator.getInterpolatedState(),
                                                                 eventYComplete);
                int index = 0;
                for (EquationsMapper secondary : expandable.getSecondaryMappers()) {
                    secondary.insertEquationData(interpolator.getInterpolatedSecondaryState(index++),
                                                 eventYComplete);
                }

                // advance all event states to current time
                for (final EventState state : eventsStates) {
                    state.stepAccepted(eventT, eventYComplete);
                    isLastStep = isLastStep || state.stop();
                }

                // handle the first part of the step, up to the event
                for (final StepHandler handler : stepHandlers) {
                    handler.handleStep(interpolator, isLastStep);
                }

                if (isLastStep) {
                    // the event asked to stop integration
                    System.arraycopy(eventYComplete, 0, y, 0, y.length);
                    return eventT;
                }

                boolean needReset = false;
                resetOccurred = false;
                needReset = currentEvent.reset(eventT, eventYComplete);
                if (needReset) {
                    // some event handler has triggered changes that
                    // invalidate the derivatives, we need to recompute them
                    interpolator.setInterpolatedTime(eventT);
                    System.arraycopy(eventYComplete, 0, y, 0, y.length);
                    computeDerivatives(eventT, y, yDot);
                    resetOccurred = true;
                    return eventT;
                }

                // prepare handling of the remaining part of the step
                previousT = eventT;
                interpolator.setSoftPreviousTime(eventT);
                interpolator.setSoftCurrentTime(currentT);

                // check if the same event occurs again in the remaining part of the step
                if (currentEvent.evaluateStep(interpolator)) {
                    // the event occurs during the current step
                    occurringEvents.add(currentEvent);
                }

            }

            // last part of the step, after the last event
            interpolator.setInterpolatedTime(currentT);
            final double[] currentY = new double[y.length];
            expandable.getPrimaryMapper().insertEquationData(interpolator.getInterpolatedState(),
                                                             currentY);
            int index = 0;
            for (EquationsMapper secondary : expandable.getSecondaryMappers()) {
                secondary.insertEquationData(interpolator.getInterpolatedSecondaryState(index++),
                                             currentY);
            }
            for (final EventState state : eventsStates) {
                state.stepAccepted(currentT, currentY);
                isLastStep = isLastStep || state.stop();
            }
            isLastStep = isLastStep || Precision.equals(currentT, tEnd, 1);

            // handle the remaining part of the step, after all events if any
            for (StepHandler handler : stepHandlers) {
                handler.handleStep(interpolator, isLastStep);
            }

            return currentT;

    }

    
    protected void sanityChecks(final ExpandableStatefulODE equations, final double t)
        throws NumberIsTooSmallException, DimensionMismatchException {

        final double threshold = 1000 * FastMath.ulp(FastMath.max(FastMath.abs(equations.getTime()),
                                                                  FastMath.abs(t)));
        final double dt = FastMath.abs(equations.getTime() - t);
        if (dt <= threshold) {
            throw new NumberIsTooSmallException(LocalizedFormats.TOO_SMALL_INTEGRATION_INTERVAL,
                                                dt, threshold, false);
        }

    }

}
