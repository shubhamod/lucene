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

import org.apache.lucene.util.hnsw.math.ode.sampling.StepInterpolator;



class EulerStepInterpolator
  extends RungeKuttaStepInterpolator {

  
  private static final long serialVersionUID = 20111120L;

  
  // CHECKSTYLE: stop RedundantModifier
  // the public modifier here is needed for serialization
  public EulerStepInterpolator() {
  }
  // CHECKSTYLE: resume RedundantModifier

  
  EulerStepInterpolator(final EulerStepInterpolator interpolator) {
    super(interpolator);
  }

  
  @Override
  protected StepInterpolator doCopy() {
    return new EulerStepInterpolator(this);
  }


  
  @Override
  protected void computeInterpolatedStateAndDerivatives(final double theta,
                                          final double oneMinusThetaH) {
      if ((previousState != null) && (theta <= 0.5)) {
          for (int i = 0; i < interpolatedState.length; ++i) {
              interpolatedState[i] = previousState[i] + theta * h * yDotK[0][i];
          }
          System.arraycopy(yDotK[0], 0, interpolatedDerivatives, 0, interpolatedDerivatives.length);
      } else {
          for (int i = 0; i < interpolatedState.length; ++i) {
              interpolatedState[i] = currentState[i] - oneMinusThetaH * yDotK[0][i];
          }
          System.arraycopy(yDotK[0], 0, interpolatedDerivatives, 0, interpolatedDerivatives.length);
      }

  }

}
