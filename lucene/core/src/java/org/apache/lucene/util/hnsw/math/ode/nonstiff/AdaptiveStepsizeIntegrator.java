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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.ode.AbstractIntegrator;
import org.apache.lucene.util.hnsw.math.ode.ExpandableStatefulODE;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public abstract class AdaptiveStepsizeIntegrator
  extends AbstractIntegrator {

    
    protected double scalAbsoluteTolerance;

    
    protected double scalRelativeTolerance;

    
    protected double[] vecAbsoluteTolerance;

    
    protected double[] vecRelativeTolerance;

    
    protected int mainSetDimension;

    
    private double initialStep;

    
    private double minStep;

    
    private double maxStep;

  
  public AdaptiveStepsizeIntegrator(final String name,
                                    final double minStep, final double maxStep,
                                    final double scalAbsoluteTolerance,
                                    final double scalRelativeTolerance) {

    super(name);
    setStepSizeControl(minStep, maxStep, scalAbsoluteTolerance, scalRelativeTolerance);
    resetInternalState();

  }

  
  public AdaptiveStepsizeIntegrator(final String name,
                                    final double minStep, final double maxStep,
                                    final double[] vecAbsoluteTolerance,
                                    final double[] vecRelativeTolerance) {

    super(name);
    setStepSizeControl(minStep, maxStep, vecAbsoluteTolerance, vecRelativeTolerance);
    resetInternalState();

  }

  
  public void setStepSizeControl(final double minimalStep, final double maximalStep,
                                 final double absoluteTolerance,
                                 final double relativeTolerance) {

      minStep     = FastMath.abs(minimalStep);
      maxStep     = FastMath.abs(maximalStep);
      initialStep = -1;

      scalAbsoluteTolerance = absoluteTolerance;
      scalRelativeTolerance = relativeTolerance;
      vecAbsoluteTolerance  = null;
      vecRelativeTolerance  = null;

  }

  
  public void setStepSizeControl(final double minimalStep, final double maximalStep,
                                 final double[] absoluteTolerance,
                                 final double[] relativeTolerance) {

      minStep     = FastMath.abs(minimalStep);
      maxStep     = FastMath.abs(maximalStep);
      initialStep = -1;

      scalAbsoluteTolerance = 0;
      scalRelativeTolerance = 0;
      vecAbsoluteTolerance  = absoluteTolerance.clone();
      vecRelativeTolerance  = relativeTolerance.clone();

  }

  
  public void setInitialStepSize(final double initialStepSize) {
    if ((initialStepSize < minStep) || (initialStepSize > maxStep)) {
      initialStep = -1.0;
    } else {
      initialStep = initialStepSize;
    }
  }

  
  @Override
  protected void sanityChecks(final ExpandableStatefulODE equations, final double t)
      throws DimensionMismatchException, NumberIsTooSmallException {

      super.sanityChecks(equations, t);

      mainSetDimension = equations.getPrimaryMapper().getDimension();

      if ((vecAbsoluteTolerance != null) && (vecAbsoluteTolerance.length != mainSetDimension)) {
          throw new DimensionMismatchException(mainSetDimension, vecAbsoluteTolerance.length);
      }

      if ((vecRelativeTolerance != null) && (vecRelativeTolerance.length != mainSetDimension)) {
          throw new DimensionMismatchException(mainSetDimension, vecRelativeTolerance.length);
      }

  }

  
  public double initializeStep(final boolean forward, final int order, final double[] scale,
                               final double t0, final double[] y0, final double[] yDot0,
                               final double[] y1, final double[] yDot1)
      throws MaxCountExceededException, DimensionMismatchException {

    if (initialStep > 0) {
      // use the user provided value
      return forward ? initialStep : -initialStep;
    }

    // very rough first guess : h = 0.01 * ||y/scale|| / ||y'/scale||
    // this guess will be used to perform an Euler step
    double ratio;
    double yOnScale2 = 0;
    double yDotOnScale2 = 0;
    for (int j = 0; j < scale.length; ++j) {
      ratio         = y0[j] / scale[j];
      yOnScale2    += ratio * ratio;
      ratio         = yDot0[j] / scale[j];
      yDotOnScale2 += ratio * ratio;
    }

    double h = ((yOnScale2 < 1.0e-10) || (yDotOnScale2 < 1.0e-10)) ?
               1.0e-6 : (0.01 * FastMath.sqrt(yOnScale2 / yDotOnScale2));
    if (! forward) {
      h = -h;
    }

    // perform an Euler step using the preceding rough guess
    for (int j = 0; j < y0.length; ++j) {
      y1[j] = y0[j] + h * yDot0[j];
    }
    computeDerivatives(t0 + h, y1, yDot1);

    // estimate the second derivative of the solution
    double yDDotOnScale = 0;
    for (int j = 0; j < scale.length; ++j) {
      ratio         = (yDot1[j] - yDot0[j]) / scale[j];
      yDDotOnScale += ratio * ratio;
    }
    yDDotOnScale = FastMath.sqrt(yDDotOnScale) / h;

    // step size is computed such that
    // h^order * max (||y'/tol||, ||y''/tol||) = 0.01
    final double maxInv2 = FastMath.max(FastMath.sqrt(yDotOnScale2), yDDotOnScale);
    final double h1 = (maxInv2 < 1.0e-15) ?
                      FastMath.max(1.0e-6, 0.001 * FastMath.abs(h)) :
                      FastMath.pow(0.01 / maxInv2, 1.0 / order);
    h = FastMath.min(100.0 * FastMath.abs(h), h1);
    h = FastMath.max(h, 1.0e-12 * FastMath.abs(t0));  // avoids cancellation when computing t1 - t0
    if (h < getMinStep()) {
      h = getMinStep();
    }
    if (h > getMaxStep()) {
      h = getMaxStep();
    }
    if (! forward) {
      h = -h;
    }

    return h;

  }

  
  protected double filterStep(final double h, final boolean forward, final boolean acceptSmall)
    throws NumberIsTooSmallException {

      double filteredH = h;
      if (FastMath.abs(h) < minStep) {
          if (acceptSmall) {
              filteredH = forward ? minStep : -minStep;
          } else {
              throw new NumberIsTooSmallException(LocalizedFormats.MINIMAL_STEPSIZE_REACHED_DURING_INTEGRATION,
                                                  FastMath.abs(h), minStep, true);
          }
      }

      if (filteredH > maxStep) {
          filteredH = maxStep;
      } else if (filteredH < -maxStep) {
          filteredH = -maxStep;
      }

      return filteredH;

  }

  
  @Override
  public abstract void integrate (ExpandableStatefulODE equations, double t)
      throws NumberIsTooSmallException, DimensionMismatchException,
             MaxCountExceededException, NoBracketingException;

  
  @Override
  public double getCurrentStepStart() {
    return stepStart;
  }

  
  protected void resetInternalState() {
    stepStart = Double.NaN;
    stepSize  = FastMath.sqrt(minStep * maxStep);
  }

  
  public double getMinStep() {
    return minStep;
  }

  
  public double getMaxStep() {
    return maxStep;
  }

}
