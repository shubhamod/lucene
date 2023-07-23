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
import org.apache.lucene.util.hnsw.math.ode.AbstractIntegrator;
import org.apache.lucene.util.hnsw.math.ode.ExpandableStatefulODE;
import org.apache.lucene.util.hnsw.math.ode.FirstOrderDifferentialEquations;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public abstract class RungeKuttaIntegrator extends AbstractIntegrator {

    
    private final double[] c;

    
    private final double[][] a;

    
    private final double[] b;

    
    private final RungeKuttaStepInterpolator prototype;

    
    private final double step;

  
  protected RungeKuttaIntegrator(final String name,
                                 final double[] c, final double[][] a, final double[] b,
                                 final RungeKuttaStepInterpolator prototype,
                                 final double step) {
    super(name);
    this.c          = c;
    this.a          = a;
    this.b          = b;
    this.prototype  = prototype;
    this.step       = FastMath.abs(step);
  }

  
  @Override
  public void integrate(final ExpandableStatefulODE equations, final double t)
      throws NumberIsTooSmallException, DimensionMismatchException,
             MaxCountExceededException, NoBracketingException {

    sanityChecks(equations, t);
    setEquations(equations);
    final boolean forward = t > equations.getTime();

    // create some internal working arrays
    final double[] y0      = equations.getCompleteState();
    final double[] y       = y0.clone();
    final int stages       = c.length + 1;
    final double[][] yDotK = new double[stages][];
    for (int i = 0; i < stages; ++i) {
      yDotK [i] = new double[y0.length];
    }
    final double[] yTmp    = y0.clone();
    final double[] yDotTmp = new double[y0.length];

    // set up an interpolator sharing the integrator arrays
    final RungeKuttaStepInterpolator interpolator = (RungeKuttaStepInterpolator) prototype.copy();
    interpolator.reinitialize(this, yTmp, yDotK, forward,
                              equations.getPrimaryMapper(), equations.getSecondaryMappers());
    interpolator.storeTime(equations.getTime());

    // set up integration control objects
    stepStart = equations.getTime();
    if (forward) {
        if (stepStart + step >= t) {
            stepSize = t - stepStart;
        } else {
            stepSize = step;
        }
    } else {
        if (stepStart - step <= t) {
            stepSize = t - stepStart;
        } else {
            stepSize = -step;
        }
    }
    initIntegration(equations.getTime(), y0, t);

    // main integration loop
    isLastStep = false;
    do {

      interpolator.shift();

      // first stage
      computeDerivatives(stepStart, y, yDotK[0]);

      // next stages
      for (int k = 1; k < stages; ++k) {

          for (int j = 0; j < y0.length; ++j) {
              double sum = a[k-1][0] * yDotK[0][j];
              for (int l = 1; l < k; ++l) {
                  sum += a[k-1][l] * yDotK[l][j];
              }
              yTmp[j] = y[j] + stepSize * sum;
          }

          computeDerivatives(stepStart + c[k-1] * stepSize, yTmp, yDotK[k]);

      }

      // estimate the state at the end of the step
      for (int j = 0; j < y0.length; ++j) {
          double sum    = b[0] * yDotK[0][j];
          for (int l = 1; l < stages; ++l) {
              sum    += b[l] * yDotK[l][j];
          }
          yTmp[j] = y[j] + stepSize * sum;
      }

      // discrete events handling
      interpolator.storeTime(stepStart + stepSize);
      System.arraycopy(yTmp, 0, y, 0, y0.length);
      System.arraycopy(yDotK[stages - 1], 0, yDotTmp, 0, y0.length);
      stepStart = acceptStep(interpolator, y, yDotTmp, t);

      if (!isLastStep) {

          // prepare next step
          interpolator.storeTime(stepStart);

          // stepsize control for next step
          final double  nextT      = stepStart + stepSize;
          final boolean nextIsLast = forward ? (nextT >= t) : (nextT <= t);
          if (nextIsLast) {
              stepSize = t - stepStart;
          }
      }

    } while (!isLastStep);

    // dispatch results
    equations.setTime(stepStart);
    equations.setCompleteState(y);

    stepStart = Double.NaN;
    stepSize  = Double.NaN;

  }

  
  public double[] singleStep(final FirstOrderDifferentialEquations equations,
                             final double t0, final double[] y0, final double t) {

      // create some internal working arrays
      final double[] y       = y0.clone();
      final int stages       = c.length + 1;
      final double[][] yDotK = new double[stages][];
      for (int i = 0; i < stages; ++i) {
          yDotK [i] = new double[y0.length];
      }
      final double[] yTmp    = y0.clone();

      // first stage
      final double h = t - t0;
      equations.computeDerivatives(t0, y, yDotK[0]);

      // next stages
      for (int k = 1; k < stages; ++k) {

          for (int j = 0; j < y0.length; ++j) {
              double sum = a[k-1][0] * yDotK[0][j];
              for (int l = 1; l < k; ++l) {
                  sum += a[k-1][l] * yDotK[l][j];
              }
              yTmp[j] = y[j] + h * sum;
          }

          equations.computeDerivatives(t0 + c[k-1] * h, yTmp, yDotK[k]);

      }

      // estimate the state at the end of the step
      for (int j = 0; j < y0.length; ++j) {
          double sum = b[0] * yDotK[0][j];
          for (int l = 1; l < stages; ++l) {
              sum += b[l] * yDotK[l][j];
          }
          y[j] += h * sum;
      }

      return y;

  }

}
