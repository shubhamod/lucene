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



class MidpointStepInterpolator
  extends RungeKuttaStepInterpolator {

  
  private static final long serialVersionUID = 20111120L;

  
  // CHECKSTYLE: stop RedundantModifier
  // the public modifier here is needed for serialization
  public MidpointStepInterpolator() {
  }
  // CHECKSTYLE: resume RedundantModifier

  
  MidpointStepInterpolator(final MidpointStepInterpolator interpolator) {
    super(interpolator);
  }

  
  @Override
  protected StepInterpolator doCopy() {
    return new MidpointStepInterpolator(this);
  }


  
  @Override
  protected void computeInterpolatedStateAndDerivatives(final double theta,
                                          final double oneMinusThetaH) {

    final double coeffDot2 = 2 * theta;
    final double coeffDot1 = 1 - coeffDot2;

    if ((previousState != null) && (theta <= 0.5)) {
        final double coeff1    = theta * oneMinusThetaH;
        final double coeff2    = theta * theta * h;
        for (int i = 0; i < interpolatedState.length; ++i) {
            final double yDot1 = yDotK[0][i];
            final double yDot2 = yDotK[1][i];
            interpolatedState[i] = previousState[i] + coeff1 * yDot1 + coeff2 * yDot2;
            interpolatedDerivatives[i] = coeffDot1 * yDot1 + coeffDot2 * yDot2;
        }
    } else {
        final double coeff1    = oneMinusThetaH * theta;
        final double coeff2    = oneMinusThetaH * (1.0 + theta);
        for (int i = 0; i < interpolatedState.length; ++i) {
            final double yDot1 = yDotK[0][i];
            final double yDot2 = yDotK[1][i];
            interpolatedState[i] = currentState[i] + coeff1 * yDot1 - coeff2 * yDot2;
            interpolatedDerivatives[i] = coeffDot1 * yDot1 + coeffDot2 * yDot2;
        }
    }

  }

}
