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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowFieldMatrix;
import org.apache.lucene.util.hnsw.math.ode.nonstiff.AdaptiveStepsizeFieldIntegrator;
import org.apache.lucene.util.hnsw.math.ode.nonstiff.DormandPrince853FieldIntegrator;
import org.apache.lucene.util.hnsw.math.ode.sampling.FieldStepHandler;
import org.apache.lucene.util.hnsw.math.ode.sampling.FieldStepInterpolator;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public abstract class MultistepFieldIntegrator<T extends RealFieldElement<T>>
    extends AdaptiveStepsizeFieldIntegrator<T> {

    
    protected T[] scaled;

    
    protected Array2DRowFieldMatrix<T> nordsieck;

    
    private FirstOrderFieldIntegrator<T> starter;

    
    private final int nSteps;

    
    private double exp;

    
    private double safety;

    
    private double minReduction;

    
    private double maxGrowth;

    
    protected MultistepFieldIntegrator(final Field<T> field, final String name,
                                       final int nSteps, final int order,
                                       final double minStep, final double maxStep,
                                       final double scalAbsoluteTolerance,
                                       final double scalRelativeTolerance)
        throws NumberIsTooSmallException {

        super(field, name, minStep, maxStep, scalAbsoluteTolerance, scalRelativeTolerance);

        if (nSteps < 2) {
            throw new NumberIsTooSmallException(
                  LocalizedFormats.INTEGRATION_METHOD_NEEDS_AT_LEAST_TWO_PREVIOUS_POINTS,
                  nSteps, 2, true);
        }

        starter = new DormandPrince853FieldIntegrator<T>(field, minStep, maxStep,
                                                         scalAbsoluteTolerance,
                                                         scalRelativeTolerance);
        this.nSteps = nSteps;

        exp = -1.0 / order;

        // set the default values of the algorithm control parameters
        setSafety(0.9);
        setMinReduction(0.2);
        setMaxGrowth(FastMath.pow(2.0, -exp));

    }

    
    protected MultistepFieldIntegrator(final Field<T> field, final String name, final int nSteps,
                                       final int order,
                                       final double minStep, final double maxStep,
                                       final double[] vecAbsoluteTolerance,
                                       final double[] vecRelativeTolerance) {
        super(field, name, minStep, maxStep, vecAbsoluteTolerance, vecRelativeTolerance);
        starter = new DormandPrince853FieldIntegrator<T>(field, minStep, maxStep,
                                                         vecAbsoluteTolerance,
                                                         vecRelativeTolerance);
        this.nSteps = nSteps;

        exp = -1.0 / order;

        // set the default values of the algorithm control parameters
        setSafety(0.9);
        setMinReduction(0.2);
        setMaxGrowth(FastMath.pow(2.0, -exp));

    }

    
    public FirstOrderFieldIntegrator<T> getStarterIntegrator() {
        return starter;
    }

    
    public void setStarterIntegrator(FirstOrderFieldIntegrator<T> starterIntegrator) {
        this.starter = starterIntegrator;
    }

    
    protected void start(final FieldExpandableODE<T> equations, final FieldODEState<T> initialState, final T t)
        throws DimensionMismatchException, NumberIsTooSmallException,
               MaxCountExceededException, NoBracketingException {

        // make sure NO user event nor user step handler is triggered,
        // this is the task of the top level integrator, not the task
        // of the starter integrator
        starter.clearEventHandlers();
        starter.clearStepHandlers();

        // set up one specific step handler to extract initial Nordsieck vector
        starter.addStepHandler(new FieldNordsieckInitializer(equations.getMapper(), (nSteps + 3) / 2));

        // start integration, expecting a InitializationCompletedMarkerException
        try {

            starter.integrate(equations, initialState, t);

            // we should not reach this step
            throw new MathIllegalStateException(LocalizedFormats.MULTISTEP_STARTER_STOPPED_EARLY);

        } catch (InitializationCompletedMarkerException icme) { // NOPMD
            // this is the expected nominal interruption of the start integrator

            // count the evaluations used by the starter
            getEvaluationsCounter().increment(starter.getEvaluations());

        }

        // remove the specific step handler
        starter.clearStepHandlers();

    }

    
    protected abstract Array2DRowFieldMatrix<T> initializeHighOrderDerivatives(final T h, final T[] t,
                                                                               final T[][] y,
                                                                               final T[][] yDot);

    
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

    
    protected void rescale(final T newStepSize) {

        final T ratio = newStepSize.divide(getStepSize());
        for (int i = 0; i < scaled.length; ++i) {
            scaled[i] = scaled[i].multiply(ratio);
        }

        final T[][] nData = nordsieck.getDataRef();
        T power = ratio;
        for (int i = 0; i < nData.length; ++i) {
            power = power.multiply(ratio);
            final T[] nDataI = nData[i];
            for (int j = 0; j < nDataI.length; ++j) {
                nDataI[j] = nDataI[j].multiply(power);
            }
        }

        setStepSize(newStepSize);

    }


    
    protected T computeStepGrowShrinkFactor(final T error) {
        return MathUtils.min(error.getField().getZero().add(maxGrowth),
                             MathUtils.max(error.getField().getZero().add(minReduction),
                                           error.pow(exp).multiply(safety)));
    }

    
    private class FieldNordsieckInitializer implements FieldStepHandler<T> {

        
        private final FieldEquationsMapper<T> mapper;

        
        private int count;

        
        private FieldODEStateAndDerivative<T> savedStart;

        
        private final T[] t;

        
        private final T[][] y;

        
        private final T[][] yDot;

        
        FieldNordsieckInitializer(final FieldEquationsMapper<T> mapper, final int nbStartPoints) {
            this.mapper = mapper;
            this.count  = 0;
            this.t      = MathArrays.buildArray(getField(), nbStartPoints);
            this.y      = MathArrays.buildArray(getField(), nbStartPoints, -1);
            this.yDot   = MathArrays.buildArray(getField(), nbStartPoints, -1);
        }

        
        public void handleStep(FieldStepInterpolator<T> interpolator, boolean isLast)
            throws MaxCountExceededException {


            if (count == 0) {
                // first step, we need to store also the point at the beginning of the step
                final FieldODEStateAndDerivative<T> prev = interpolator.getPreviousState();
                savedStart  = prev;
                t[count]    = prev.getTime();
                y[count]    = mapper.mapState(prev);
                yDot[count] = mapper.mapDerivative(prev);
            }

            // store the point at the end of the step
            ++count;
            final FieldODEStateAndDerivative<T> curr = interpolator.getCurrentState();
            t[count]    = curr.getTime();
            y[count]    = mapper.mapState(curr);
            yDot[count] = mapper.mapDerivative(curr);

            if (count == t.length - 1) {

                // this was the last point we needed, we can compute the derivatives
                setStepSize(t[t.length - 1].subtract(t[0]).divide(t.length - 1));

                // first scaled derivative
                scaled = MathArrays.buildArray(getField(), yDot[0].length);
                for (int j = 0; j < scaled.length; ++j) {
                    scaled[j] = yDot[0][j].multiply(getStepSize());
                }

                // higher order derivatives
                nordsieck = initializeHighOrderDerivatives(getStepSize(), t, y, yDot);

                // stop the integrator now that all needed steps have been handled
                setStepStart(savedStart);
                throw new InitializationCompletedMarkerException();

            }

        }

        
        public void init(final FieldODEStateAndDerivative<T> initialState, T finalTime) {
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
