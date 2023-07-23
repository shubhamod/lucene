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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.ode.AbstractFieldIntegrator;
import org.apache.lucene.util.hnsw.math.ode.FieldEquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.FieldODEState;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;



public abstract class AdaptiveStepsizeFieldIntegrator<T extends RealFieldElement<T>>
    extends AbstractFieldIntegrator<T> {

    
    protected double scalAbsoluteTolerance;

    
    protected double scalRelativeTolerance;

    
    protected double[] vecAbsoluteTolerance;

    
    protected double[] vecRelativeTolerance;

    
    protected int mainSetDimension;

    
    private T initialStep;

    
    private T minStep;

    
    private T maxStep;

    
    public AdaptiveStepsizeFieldIntegrator(final Field<T> field, final String name,
                                           final double minStep, final double maxStep,
                                           final double scalAbsoluteTolerance,
                                           final double scalRelativeTolerance) {

        super(field, name);
        setStepSizeControl(minStep, maxStep, scalAbsoluteTolerance, scalRelativeTolerance);
        resetInternalState();

    }

    
    public AdaptiveStepsizeFieldIntegrator(final Field<T> field, final String name,
                                           final double minStep, final double maxStep,
                                           final double[] vecAbsoluteTolerance,
                                           final double[] vecRelativeTolerance) {

        super(field, name);
        setStepSizeControl(minStep, maxStep, vecAbsoluteTolerance, vecRelativeTolerance);
        resetInternalState();

    }

    
    public void setStepSizeControl(final double minimalStep, final double maximalStep,
                                   final double absoluteTolerance,
                                   final double relativeTolerance) {

        minStep     = getField().getZero().add(FastMath.abs(minimalStep));
        maxStep     = getField().getZero().add(FastMath.abs(maximalStep));
        initialStep = getField().getOne().negate();

        scalAbsoluteTolerance = absoluteTolerance;
        scalRelativeTolerance = relativeTolerance;
        vecAbsoluteTolerance  = null;
        vecRelativeTolerance  = null;

    }

    
    public void setStepSizeControl(final double minimalStep, final double maximalStep,
                                   final double[] absoluteTolerance,
                                   final double[] relativeTolerance) {

        minStep     = getField().getZero().add(FastMath.abs(minimalStep));
        maxStep     = getField().getZero().add(FastMath.abs(maximalStep));
        initialStep = getField().getOne().negate();

        scalAbsoluteTolerance = 0;
        scalRelativeTolerance = 0;
        vecAbsoluteTolerance  = absoluteTolerance.clone();
        vecRelativeTolerance  = relativeTolerance.clone();

    }

    
    public void setInitialStepSize(final T initialStepSize) {
        if (initialStepSize.subtract(minStep).getReal() < 0 ||
            initialStepSize.subtract(maxStep).getReal() > 0) {
            initialStep = getField().getOne().negate();
        } else {
            initialStep = initialStepSize;
        }
    }

    
    @Override
    protected void sanityChecks(final FieldODEState<T> eqn, final T t)
        throws DimensionMismatchException, NumberIsTooSmallException {

        super.sanityChecks(eqn, t);

        mainSetDimension = eqn.getStateDimension();

        if (vecAbsoluteTolerance != null && vecAbsoluteTolerance.length != mainSetDimension) {
            throw new DimensionMismatchException(mainSetDimension, vecAbsoluteTolerance.length);
        }

        if (vecRelativeTolerance != null && vecRelativeTolerance.length != mainSetDimension) {
            throw new DimensionMismatchException(mainSetDimension, vecRelativeTolerance.length);
        }

    }

    
    public T initializeStep(final boolean forward, final int order, final T[] scale,
                            final FieldODEStateAndDerivative<T> state0,
                            final FieldEquationsMapper<T> mapper)
        throws MaxCountExceededException, DimensionMismatchException {

        if (initialStep.getReal() > 0) {
            // use the user provided value
            return forward ? initialStep : initialStep.negate();
        }

        // very rough first guess : h = 0.01 * ||y/scale|| / ||y'/scale||
        // this guess will be used to perform an Euler step
        final T[] y0    = mapper.mapState(state0);
        final T[] yDot0 = mapper.mapDerivative(state0);
        T yOnScale2    = getField().getZero();
        T yDotOnScale2 = getField().getZero();
        for (int j = 0; j < scale.length; ++j) {
            final T ratio    = y0[j].divide(scale[j]);
            yOnScale2        = yOnScale2.add(ratio.multiply(ratio));
            final T ratioDot = yDot0[j].divide(scale[j]);
            yDotOnScale2     = yDotOnScale2.add(ratioDot.multiply(ratioDot));
        }

        T h = (yOnScale2.getReal() < 1.0e-10 || yDotOnScale2.getReal() < 1.0e-10) ?
              getField().getZero().add(1.0e-6) :
              yOnScale2.divide(yDotOnScale2).sqrt().multiply(0.01);
        if (! forward) {
            h = h.negate();
        }

        // perform an Euler step using the preceding rough guess
        final T[] y1 = MathArrays.buildArray(getField(), y0.length);
        for (int j = 0; j < y0.length; ++j) {
            y1[j] = y0[j].add(yDot0[j].multiply(h));
        }
        final T[] yDot1 = computeDerivatives(state0.getTime().add(h), y1);

        // estimate the second derivative of the solution
        T yDDotOnScale = getField().getZero();
        for (int j = 0; j < scale.length; ++j) {
            final T ratioDotDot = yDot1[j].subtract(yDot0[j]).divide(scale[j]);
            yDDotOnScale = yDDotOnScale.add(ratioDotDot.multiply(ratioDotDot));
        }
        yDDotOnScale = yDDotOnScale.sqrt().divide(h);

        // step size is computed such that
        // h^order * max (||y'/tol||, ||y''/tol||) = 0.01
        final T maxInv2 = MathUtils.max(yDotOnScale2.sqrt(), yDDotOnScale);
        final T h1 = maxInv2.getReal() < 1.0e-15 ?
                     MathUtils.max(getField().getZero().add(1.0e-6), h.abs().multiply(0.001)) :
                     maxInv2.multiply(100).reciprocal().pow(1.0 / order);
        h = MathUtils.min(h.abs().multiply(100), h1);
        h = MathUtils.max(h, state0.getTime().abs().multiply(1.0e-12));  // avoids cancellation when computing t1 - t0
        h = MathUtils.max(minStep, MathUtils.min(maxStep, h));
        if (! forward) {
            h = h.negate();
        }

        return h;

    }

    
    protected T filterStep(final T h, final boolean forward, final boolean acceptSmall)
        throws NumberIsTooSmallException {

        T filteredH = h;
        if (h.abs().subtract(minStep).getReal() < 0) {
            if (acceptSmall) {
                filteredH = forward ? minStep : minStep.negate();
            } else {
                throw new NumberIsTooSmallException(LocalizedFormats.MINIMAL_STEPSIZE_REACHED_DURING_INTEGRATION,
                                                    h.abs().getReal(), minStep.getReal(), true);
            }
        }

        if (filteredH.subtract(maxStep).getReal() > 0) {
            filteredH = maxStep;
        } else if (filteredH.add(maxStep).getReal() < 0) {
            filteredH = maxStep.negate();
        }

        return filteredH;

    }

    
    protected void resetInternalState() {
        setStepStart(null);
        setStepSize(minStep.multiply(maxStep).sqrt());
    }

    
    public T getMinStep() {
        return minStep;
    }

    
    public T getMaxStep() {
        return maxStep;
    }

}
