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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.ode.sampling.StepHandler;
import org.apache.lucene.util.hnsw.math.ode.sampling.StepInterpolator;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class ContinuousOutputModel
  implements StepHandler, Serializable {

    
    private static final long serialVersionUID = -1417964919405031606L;

    
    private double initialTime;

    
    private double finalTime;

    
    private boolean forward;

    
    private int index;

    
    private List<StepInterpolator> steps;

  
  public ContinuousOutputModel() {
    steps = new ArrayList<StepInterpolator>();
    initialTime = Double.NaN;
    finalTime   = Double.NaN;
    forward     = true;
    index       = 0;
  }

  
  public void append(final ContinuousOutputModel model)
    throws MathIllegalArgumentException, MaxCountExceededException {

    if (model.steps.size() == 0) {
      return;
    }

    if (steps.size() == 0) {
      initialTime = model.initialTime;
      forward     = model.forward;
    } else {

      if (getInterpolatedState().length != model.getInterpolatedState().length) {
          throw new DimensionMismatchException(model.getInterpolatedState().length,
                                               getInterpolatedState().length);
      }

      if (forward ^ model.forward) {
          throw new MathIllegalArgumentException(LocalizedFormats.PROPAGATION_DIRECTION_MISMATCH);
      }

      final StepInterpolator lastInterpolator = steps.get(index);
      final double current  = lastInterpolator.getCurrentTime();
      final double previous = lastInterpolator.getPreviousTime();
      final double step = current - previous;
      final double gap = model.getInitialTime() - current;
      if (FastMath.abs(gap) > 1.0e-3 * FastMath.abs(step)) {
        throw new MathIllegalArgumentException(LocalizedFormats.HOLE_BETWEEN_MODELS_TIME_RANGES,
                                               FastMath.abs(gap));
      }

    }

    for (StepInterpolator interpolator : model.steps) {
      steps.add(interpolator.copy());
    }

    index = steps.size() - 1;
    finalTime = (steps.get(index)).getCurrentTime();

  }

  
  public void init(double t0, double[] y0, double t) {
    initialTime = Double.NaN;
    finalTime   = Double.NaN;
    forward     = true;
    index       = 0;
    steps.clear();
  }

  
  public void handleStep(final StepInterpolator interpolator, final boolean isLast)
      throws MaxCountExceededException {

    if (steps.size() == 0) {
      initialTime = interpolator.getPreviousTime();
      forward     = interpolator.isForward();
    }

    steps.add(interpolator.copy());

    if (isLast) {
      finalTime = interpolator.getCurrentTime();
      index     = steps.size() - 1;
    }

  }

  
  public double getInitialTime() {
    return initialTime;
  }

  
  public double getFinalTime() {
    return finalTime;
  }

  
  public double getInterpolatedTime() {
    return steps.get(index).getInterpolatedTime();
  }

  
  public void setInterpolatedTime(final double time) {

      // initialize the search with the complete steps table
      int iMin = 0;
      final StepInterpolator sMin = steps.get(iMin);
      double tMin = 0.5 * (sMin.getPreviousTime() + sMin.getCurrentTime());

      int iMax = steps.size() - 1;
      final StepInterpolator sMax = steps.get(iMax);
      double tMax = 0.5 * (sMax.getPreviousTime() + sMax.getCurrentTime());

      // handle points outside of the integration interval
      // or in the first and last step
      if (locatePoint(time, sMin) <= 0) {
        index = iMin;
        sMin.setInterpolatedTime(time);
        return;
      }
      if (locatePoint(time, sMax) >= 0) {
        index = iMax;
        sMax.setInterpolatedTime(time);
        return;
      }

      // reduction of the table slice size
      while (iMax - iMin > 5) {

        // use the last estimated index as the splitting index
        final StepInterpolator si = steps.get(index);
        final int location = locatePoint(time, si);
        if (location < 0) {
          iMax = index;
          tMax = 0.5 * (si.getPreviousTime() + si.getCurrentTime());
        } else if (location > 0) {
          iMin = index;
          tMin = 0.5 * (si.getPreviousTime() + si.getCurrentTime());
        } else {
          // we have found the target step, no need to continue searching
          si.setInterpolatedTime(time);
          return;
        }

        // compute a new estimate of the index in the reduced table slice
        final int iMed = (iMin + iMax) / 2;
        final StepInterpolator sMed = steps.get(iMed);
        final double tMed = 0.5 * (sMed.getPreviousTime() + sMed.getCurrentTime());

        if ((FastMath.abs(tMed - tMin) < 1e-6) || (FastMath.abs(tMax - tMed) < 1e-6)) {
          // too close to the bounds, we estimate using a simple dichotomy
          index = iMed;
        } else {
          // estimate the index using a reverse quadratic polynom
          // (reverse means we have i = P(t), thus allowing to simply
          // compute index = P(time) rather than solving a quadratic equation)
          final double d12 = tMax - tMed;
          final double d23 = tMed - tMin;
          final double d13 = tMax - tMin;
          final double dt1 = time - tMax;
          final double dt2 = time - tMed;
          final double dt3 = time - tMin;
          final double iLagrange = ((dt2 * dt3 * d23) * iMax -
                                    (dt1 * dt3 * d13) * iMed +
                                    (dt1 * dt2 * d12) * iMin) /
                                   (d12 * d23 * d13);
          index = (int) FastMath.rint(iLagrange);
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
      while ((index <= iMax) && (locatePoint(time, steps.get(index)) > 0)) {
        ++index;
      }

      steps.get(index).setInterpolatedTime(time);

  }

  
  public double[] getInterpolatedState() throws MaxCountExceededException {
    return steps.get(index).getInterpolatedState();
  }

  
  public double[] getInterpolatedDerivatives() throws MaxCountExceededException {
    return steps.get(index).getInterpolatedDerivatives();
  }

  
  public double[] getInterpolatedSecondaryState(final int secondaryStateIndex)
    throws MaxCountExceededException {
    return steps.get(index).getInterpolatedSecondaryState(secondaryStateIndex);
  }

  
  public double[] getInterpolatedSecondaryDerivatives(final int secondaryStateIndex)
    throws MaxCountExceededException {
    return steps.get(index).getInterpolatedSecondaryDerivatives(secondaryStateIndex);
  }

  
  private int locatePoint(final double time, final StepInterpolator interval) {
    if (forward) {
      if (time < interval.getPreviousTime()) {
        return -1;
      } else if (time > interval.getCurrentTime()) {
        return +1;
      } else {
        return 0;
      }
    }
    if (time > interval.getPreviousTime()) {
      return -1;
    } else if (time < interval.getCurrentTime()) {
      return +1;
    } else {
      return 0;
    }
  }

}
