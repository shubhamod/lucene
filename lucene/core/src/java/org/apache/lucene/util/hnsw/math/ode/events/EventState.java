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

package org.apache.lucene.util.hnsw.math.ode.events;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.solvers.AllowedSolution;
import org.apache.lucene.util.hnsw.math.analysis.solvers.BracketedUnivariateSolver;
import org.apache.lucene.util.hnsw.math.analysis.solvers.PegasusSolver;
import org.apache.lucene.util.hnsw.math.analysis.solvers.UnivariateSolver;
import org.apache.lucene.util.hnsw.math.analysis.solvers.UnivariateSolverUtils;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.ode.EquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.ExpandableStatefulODE;
import org.apache.lucene.util.hnsw.math.ode.sampling.StepInterpolator;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class EventState {

    
    private final EventHandler handler;

    
    private final double maxCheckInterval;

    
    private final double convergence;

    
    private final int maxIterationCount;

    
    private ExpandableStatefulODE expandable;

    
    private double t0;

    
    private double g0;

    
    private boolean g0Positive;

    
    private boolean pendingEvent;

    
    private double pendingEventTime;

    
    private double previousEventTime;

    
    private boolean forward;

    
    private boolean increasing;

    
    private EventHandler.Action nextAction;

    
    private final UnivariateSolver solver;

    
    public EventState(final EventHandler handler, final double maxCheckInterval,
                      final double convergence, final int maxIterationCount,
                      final UnivariateSolver solver) {
        this.handler           = handler;
        this.maxCheckInterval  = maxCheckInterval;
        this.convergence       = FastMath.abs(convergence);
        this.maxIterationCount = maxIterationCount;
        this.solver            = solver;

        // some dummy values ...
        expandable        = null;
        t0                = Double.NaN;
        g0                = Double.NaN;
        g0Positive        = true;
        pendingEvent      = false;
        pendingEventTime  = Double.NaN;
        previousEventTime = Double.NaN;
        increasing        = true;
        nextAction        = EventHandler.Action.CONTINUE;

    }

    
    public EventHandler getEventHandler() {
        return handler;
    }

    
    public void setExpandable(final ExpandableStatefulODE expandable) {
        this.expandable = expandable;
    }

    
    public double getMaxCheckInterval() {
        return maxCheckInterval;
    }

    
    public double getConvergence() {
        return convergence;
    }

    
    public int getMaxIterationCount() {
        return maxIterationCount;
    }

    
    public void reinitializeBegin(final StepInterpolator interpolator)
        throws MaxCountExceededException {

        t0 = interpolator.getPreviousTime();
        interpolator.setInterpolatedTime(t0);
        g0 = handler.g(t0, getCompleteState(interpolator));
        if (g0 == 0) {
            // excerpt from MATH-421 issue:
            // If an ODE solver is setup with an EventHandler that return STOP
            // when the even is triggered, the integrator stops (which is exactly
            // the expected behavior). If however the user wants to restart the
            // solver from the final state reached at the event with the same
            // configuration (expecting the event to be triggered again at a
            // later time), then the integrator may fail to start. It can get stuck
            // at the previous event. The use case for the bug MATH-421 is fairly
            // general, so events occurring exactly at start in the first step should
            // be ignored.

            // extremely rare case: there is a zero EXACTLY at interval start
            // we will use the sign slightly after step beginning to force ignoring this zero
            final double epsilon = FastMath.max(solver.getAbsoluteAccuracy(),
                                                FastMath.abs(solver.getRelativeAccuracy() * t0));
            final double tStart = t0 + 0.5 * epsilon;
            interpolator.setInterpolatedTime(tStart);
            g0 = handler.g(tStart, getCompleteState(interpolator));
        }
        g0Positive = g0 >= 0;

    }

    
    private double[] getCompleteState(final StepInterpolator interpolator) {

        final double[] complete = new double[expandable.getTotalDimension()];

        expandable.getPrimaryMapper().insertEquationData(interpolator.getInterpolatedState(),
                                                         complete);
        int index = 0;
        for (EquationsMapper secondary : expandable.getSecondaryMappers()) {
            secondary.insertEquationData(interpolator.getInterpolatedSecondaryState(index++),
                                         complete);
        }

        return complete;

    }

    
    public boolean evaluateStep(final StepInterpolator interpolator)
        throws MaxCountExceededException, NoBracketingException {

        try {
            forward = interpolator.isForward();
            final double t1 = interpolator.getCurrentTime();
            final double dt = t1 - t0;
            if (FastMath.abs(dt) < convergence) {
                // we cannot do anything on such a small step, don't trigger any events
                return false;
            }
            final int    n = FastMath.max(1, (int) FastMath.ceil(FastMath.abs(dt) / maxCheckInterval));
            final double h = dt / n;

            final UnivariateFunction f = new UnivariateFunction() {
                
                public double value(final double t) throws LocalMaxCountExceededException {
                    try {
                        interpolator.setInterpolatedTime(t);
                        return handler.g(t, getCompleteState(interpolator));
                    } catch (MaxCountExceededException mcee) {
                        throw new LocalMaxCountExceededException(mcee);
                    }
                }
            };

            double ta = t0;
            double ga = g0;
            for (int i = 0; i < n; ++i) {

                // evaluate handler value at the end of the substep
                final double tb = (i == n - 1) ? t1 : t0 + (i + 1) * h;
                interpolator.setInterpolatedTime(tb);
                final double gb = handler.g(tb, getCompleteState(interpolator));

                // check events occurrence
                if (g0Positive ^ (gb >= 0)) {
                    // there is a sign change: an event is expected during this step

                    // variation direction, with respect to the integration direction
                    increasing = gb >= ga;

                    // find the event time making sure we select a solution just at or past the exact root
                    final double root;
                    if (solver instanceof BracketedUnivariateSolver<?>) {
                        @SuppressWarnings("unchecked")
                        BracketedUnivariateSolver<UnivariateFunction> bracketing =
                                (BracketedUnivariateSolver<UnivariateFunction>) solver;
                        root = forward ?
                               bracketing.solve(maxIterationCount, f, ta, tb, AllowedSolution.RIGHT_SIDE) :
                               bracketing.solve(maxIterationCount, f, tb, ta, AllowedSolution.LEFT_SIDE);
                    } else {
                        final double baseRoot = forward ?
                                                solver.solve(maxIterationCount, f, ta, tb) :
                                                solver.solve(maxIterationCount, f, tb, ta);
                        final int remainingEval = maxIterationCount - solver.getEvaluations();
                        BracketedUnivariateSolver<UnivariateFunction> bracketing =
                                new PegasusSolver(solver.getRelativeAccuracy(), solver.getAbsoluteAccuracy());
                        root = forward ?
                               UnivariateSolverUtils.forceSide(remainingEval, f, bracketing,
                                                                   baseRoot, ta, tb, AllowedSolution.RIGHT_SIDE) :
                               UnivariateSolverUtils.forceSide(remainingEval, f, bracketing,
                                                                   baseRoot, tb, ta, AllowedSolution.LEFT_SIDE);
                    }

                    if ((!Double.isNaN(previousEventTime)) &&
                        (FastMath.abs(root - ta) <= convergence) &&
                        (FastMath.abs(root - previousEventTime) <= convergence)) {
                        // we have either found nothing or found (again ?) a past event,
                        // retry the substep excluding this value, and taking care to have the
                        // required sign in case the g function is noisy around its zero and
                        // crosses the axis several times
                        do {
                            ta = forward ? ta + convergence : ta - convergence;
                            ga = f.value(ta);
                        } while ((g0Positive ^ (ga >= 0)) && (forward ^ (ta >= tb)));

                        if (forward ^ (ta >= tb)) {
                            // we were able to skip this spurious root
                            --i;
                        } else {
                            // we can't avoid this root before the end of the step,
                            // we have to handle it despite it is close to the former one
                            // maybe we have two very close roots
                            pendingEventTime = root;
                            pendingEvent = true;
                            return true;
                        }
                    } else if (Double.isNaN(previousEventTime) ||
                               (FastMath.abs(previousEventTime - root) > convergence)) {
                        pendingEventTime = root;
                        pendingEvent = true;
                        return true;
                    } else {
                        // no sign change: there is no event for now
                        ta = tb;
                        ga = gb;
                    }

                } else {
                    // no sign change: there is no event for now
                    ta = tb;
                    ga = gb;
                }

            }

            // no event during the whole step
            pendingEvent     = false;
            pendingEventTime = Double.NaN;
            return false;

        } catch (LocalMaxCountExceededException lmcee) {
            throw lmcee.getException();
        }

    }

    
    public double getEventTime() {
        return pendingEvent ?
               pendingEventTime :
               (forward ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY);
    }

    
    public void stepAccepted(final double t, final double[] y) {

        t0 = t;
        g0 = handler.g(t, y);

        if (pendingEvent && (FastMath.abs(pendingEventTime - t) <= convergence)) {
            // force the sign to its value "just after the event"
            previousEventTime = t;
            g0Positive        = increasing;
            nextAction        = handler.eventOccurred(t, y, !(increasing ^ forward));
        } else {
            g0Positive = g0 >= 0;
            nextAction = EventHandler.Action.CONTINUE;
        }
    }

    
    public boolean stop() {
        return nextAction == EventHandler.Action.STOP;
    }

    
    public boolean reset(final double t, final double[] y) {

        if (!(pendingEvent && (FastMath.abs(pendingEventTime - t) <= convergence))) {
            return false;
        }

        if (nextAction == EventHandler.Action.RESET_STATE) {
            handler.resetState(t, y);
        }
        pendingEvent      = false;
        pendingEventTime  = Double.NaN;

        return (nextAction == EventHandler.Action.RESET_STATE) ||
               (nextAction == EventHandler.Action.RESET_DERIVATIVES);

    }

    
    private static class LocalMaxCountExceededException extends RuntimeException {

        
        private static final long serialVersionUID = 20120901L;

        
        private final MaxCountExceededException wrapped;

        
        LocalMaxCountExceededException(final MaxCountExceededException exception) {
            wrapped = exception;
        }

        
        public MaxCountExceededException getException() {
            return wrapped;
        }

    }

}
