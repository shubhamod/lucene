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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.ode.nonstiff.AdaptiveStepsizeIntegrator;
import org.apache.lucene.util.hnsw.math.ode.nonstiff.DormandPrince853Integrator;
import org.apache.lucene.util.hnsw.math.ode.sampling.StepHandler;
import org.apache.lucene.util.hnsw.math.ode.sampling.StepInterpolator;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public abstract class MultistepIntegrator extends AdaptiveStepsizeIntegrator {

    
    protected double[] scaled;

    
    protected Array2DRowRealMatrix nordsieck;

    
    private FirstOrderIntegrator starter;

    
    private final int nSteps;

    
    private double exp;

    
    private double safety;

    
    private double minReduction;

    
    private double maxGrowth;

    
    protected MultistepIntegrator(final String name, final int nSteps,
                                  final int order,
                                  final double minStep, final double maxStep,
                                  final double scalAbsoluteTolerance,
                                  final double scalRelativeTolerance)
        throws NumberIsTooSmallException {

        super(name, minStep, maxStep, scalAbsoluteTolerance, scalRelativeTolerance);

        if (nSteps < 2) {
            throw new NumberIsTooSmallException(
                  LocalizedFormats.INTEGRATION_METHOD_NEEDS_AT_LEAST_TWO_PREVIOUS_POINTS,
                  nSteps, 2, true);
        }

        starter = new DormandPrince853Integrator(minStep, maxStep,
                                                 scalAbsoluteTolerance,
                                                 scalRelativeTolerance);
        this.nSteps = nSteps;

        exp = -1.0 / order;

        // set the default values of the algorithm control parameters
        setSafety(0.9);
        setMinReduction(0.2);
        setMaxGrowth(FastMath.pow(2.0, -exp));

    }

    
    protected MultistepIntegrator(final String name, final int nSteps,
                                  final int order,
                                  final double minStep, final double maxStep,
                                  final double[] vecAbsoluteTolerance,
                                  final double[] vecRelativeTolerance) {
        super(name, minStep, maxStep, vecAbsoluteTolerance, vecRelativeTolerance);
        starter = new DormandPrince853Integrator(minStep, maxStep,
                                                 vecAbsoluteTolerance,
                                                 vecRelativeTolerance);
        this.nSteps = nSteps;

        exp = -1.0 / order;

        // set the default values of the algorithm control parameters
        setSafety(0.9);
        setMinReduction(0.2);
        setMaxGrowth(FastMath.pow(2.0, -exp));

    }

    
    public ODEIntegrator getStarterIntegrator() {
        return starter;
    }

    
    public void setStarterIntegrator(FirstOrderIntegrator starterIntegrator) {
        this.starter = starterIntegrator;
    }

    
    protected void start(final double t0, final double[] y0, final double t)
        throws DimensionMismatchException, NumberIsTooSmallException,
               MaxCountExceededException, NoBracketingException {

        // make sure NO user event nor user step handler is triggered,
        // this is the task of the top level integrator, not the task
        // of the starter integrator
        starter.clearEventHandlers();
        starter.clearStepHandlers();

        // set up one specific step handler to extract initial Nordsieck vector
        starter.addStepHandler(new NordsieckInitializer((nSteps + 3) / 2, y0.length));

        // start integration, expecting a InitializationCompletedMarkerException
        try {

            if (starter instanceof AbstractIntegrator) {
                ((AbstractIntegrator) starter).integrate(getExpandable(), t);
            } else {
                starter.integrate(new FirstOrderDifferentialEquations() {

                    
                    public int getDimension() {
                        return getExpandable().getTotalDimension();
                    }

                    
                    public void computeDerivatives(double t, double[] y, double[] yDot) {
                        getExpandable().computeDerivatives(t, y, yDot);
                    }

                }, t0, y0, t, new double[y0.length]);
            }

            // we should not reach this step
            throw new MathIllegalStateException(LocalizedFormats.MULTISTEP_STARTER_STOPPED_EARLY);

        } catch (InitializationCompletedMarkerException icme) { // NOPMD
            // this is the expected nominal interruption of the start integrator

            // count the evaluations used by the starter
            getCounter().increment(starter.getEvaluations());

        }

        // remove the specific step handler
        starter.clearStepHandlers();

    }

    
    protected abstract Array2DRowRealMatrix initializeHighOrderDerivatives(final double h, final double[] t,
                                                                           final double[][] y,
                                                                           final double[][] yDot);

    
    public double getMinReduction() {
        return minReduction;
    }

    
    public void setMinReduction(final double minReduction) {
        this.minReduction = minReduction;
    }

    
    public double getMaxGrowth() {
        return maxGrowth;
    }

    
    public void setMaxGrowth(final double maxGrowth) {
        this.maxGrowth = maxGrowth;
    }

    
    public double getSafety() {
      return safety;
    }

    
    public void setSafety(final double safety) {
      this.safety = safety;
    }

    
    public int getNSteps() {
      return nSteps;
    }

    
    protected double computeStepGrowShrinkFactor(final double error) {
        return FastMath.min(maxGrowth, FastMath.max(minReduction, safety * FastMath.pow(error, exp)));
    }

    
    @Deprecated
    public interface NordsieckTransformer {
        
        Array2DRowRealMatrix initializeHighOrderDerivatives(final double h, final double[] t,
                                                            final double[][] y,
                                                            final double[][] yDot);
    }

    
    private class NordsieckInitializer implements StepHandler {

        
        private int count;

        
        private final double[] t;

        
        private final double[][] y;

        
        private final double[][] yDot;

        
        NordsieckInitializer(final int nbStartPoints, final int n) {
            this.count = 0;
            this.t     = new double[nbStartPoints];
            this.y     = new double[nbStartPoints][n];
            this.yDot  = new double[nbStartPoints][n];
        }

        
        public void handleStep(StepInterpolator interpolator, boolean isLast)
            throws MaxCountExceededException {

            final double prev = interpolator.getPreviousTime();
            final double curr = interpolator.getCurrentTime();

            if (count == 0) {
                // first step, we need to store also the point at the beginning of the step
                interpolator.setInterpolatedTime(prev);
                t[0] = prev;
                final ExpandableStatefulODE expandable = getExpandable();
                final EquationsMapper primary = expandable.getPrimaryMapper();
                primary.insertEquationData(interpolator.getInterpolatedState(), y[count]);
                primary.insertEquationData(interpolator.getInterpolatedDerivatives(), yDot[count]);
                int index = 0;
                for (final EquationsMapper secondary : expandable.getSecondaryMappers()) {
                    secondary.insertEquationData(interpolator.getInterpolatedSecondaryState(index), y[count]);
                    secondary.insertEquationData(interpolator.getInterpolatedSecondaryDerivatives(index), yDot[count]);
                    ++index;
                }
            }

            // store the point at the end of the step
            ++count;
            interpolator.setInterpolatedTime(curr);
            t[count] = curr;

            final ExpandableStatefulODE expandable = getExpandable();
            final EquationsMapper primary = expandable.getPrimaryMapper();
            primary.insertEquationData(interpolator.getInterpolatedState(), y[count]);
            primary.insertEquationData(interpolator.getInterpolatedDerivatives(), yDot[count]);
            int index = 0;
            for (final EquationsMapper secondary : expandable.getSecondaryMappers()) {
                secondary.insertEquationData(interpolator.getInterpolatedSecondaryState(index), y[count]);
                secondary.insertEquationData(interpolator.getInterpolatedSecondaryDerivatives(index), yDot[count]);
                ++index;
            }

            if (count == t.length - 1) {

                // this was the last point we needed, we can compute the derivatives
                stepStart = t[0];
                stepSize  = (t[t.length - 1] - t[0]) / (t.length - 1);

                // first scaled derivative
                scaled = yDot[0].clone();
                for (int j = 0; j < scaled.length; ++j) {
                    scaled[j] *= stepSize;
                }

                // higher order derivatives
                nordsieck = initializeHighOrderDerivatives(stepSize, t, y, yDot);

                // stop the integrator now that all needed steps have been handled
                throw new InitializationCompletedMarkerException();

            }

        }

        
        public void init(double t0, double[] y0, double time) {
            // nothing to do
        }

    }

    
    private static class InitializationCompletedMarkerException
        extends RuntimeException {

        
        private static final long serialVersionUID = -1914085471038046418L;

        
        InitializationCompletedMarkerException() {
            super((Throwable) null);
        }

    }

}
