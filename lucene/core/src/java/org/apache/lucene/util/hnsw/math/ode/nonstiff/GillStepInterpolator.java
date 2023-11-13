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
import org.apache.lucene.util.hnsw.math.util.FastMath;



class GillStepInterpolator
  extends RungeKuttaStepInterpolator {

    
    private static final double ONE_MINUS_INV_SQRT_2 = 1 - FastMath.sqrt(0.5);

    
    private static final double ONE_PLUS_INV_SQRT_2 = 1 + FastMath.sqrt(0.5);

    
    private static final long serialVersionUID = 20111120L;

  
  // CHECKSTYLE: stop RedundantModifier
  // the public modifier here is needed for serialization
  public GillStepInterpolator() {
  }
  // CHECKSTYLE: resume RedundantModifier

  
  GillStepInterpolator(final GillStepInterpolator interpolator) {
    super(interpolator);
  }

  
  @Override
  protected StepInterpolator doCopy() {
    return new GillStepInterpolator(this);
  }


  
  @Override
  protected void computeInterpolatedStateAndDerivatives(final double theta,
                                          final double oneMinusThetaH) {

    final double twoTheta   = 2 * theta;
    final double fourTheta2 = twoTheta * twoTheta;
    final double coeffDot1  = theta * (twoTheta - 3) + 1;
    final double cDot23     = twoTheta * (1 - theta);
    final double coeffDot2  = cDot23  * ONE_MINUS_INV_SQRT_2;
    final double coeffDot3  = cDot23  * ONE_PLUS_INV_SQRT_2;
    final double coeffDot4  = theta * (twoTheta - 1);

    if ((previousState != null) && (theta <= 0.5)) {
        final double s         = theta * h / 6.0;
        final double c23       = s * (6 * theta - fourTheta2);
        final double coeff1    = s * (6 - 9 * theta + fourTheta2);
        final double coeff2    = c23  * ONE_MINUS_INV_SQRT_2;
        final double coeff3    = c23  * ONE_PLUS_INV_SQRT_2;
        final double coeff4    = s * (-3 * theta + fourTheta2);
        for (int i = 0; i < interpolatedState.length; ++i) {
            final double yDot1 = yDotK[0][i];
            final double yDot2 = yDotK[1][i];
            final double yDot3 = yDotK[2][i];
            final double yDot4 = yDotK[3][i];
            interpolatedState[i] =
                    previousState[i] + coeff1 * yDot1 + coeff2 * yDot2 + coeff3 * yDot3 + coeff4 * yDot4;
            interpolatedDerivatives[i] =
                    coeffDot1 * yDot1 + coeffDot2 * yDot2 + coeffDot3 * yDot3 + coeffDot4 * yDot4;
        }
    } else {
        final double s      = oneMinusThetaH / 6.0;
        final double c23    = s * (2 + twoTheta - fourTheta2);
        final double coeff1 = s * (1 - 5 * theta + fourTheta2);
        final double coeff2 = c23  * ONE_MINUS_INV_SQRT_2;
        final double coeff3 = c23  * ONE_PLUS_INV_SQRT_2;
        final double coeff4 = s * (1 + theta + fourTheta2);
        for (int i = 0; i < interpolatedState.length; ++i) {
            final double yDot1 = yDotK[0][i];
            final double yDot2 = yDotK[1][i];
            final double yDot3 = yDotK[2][i];
            final double yDot4 = yDotK[3][i];
            interpolatedState[i] =
                    currentState[i] - coeff1 * yDot1 - coeff2 * yDot2 - coeff3 * yDot3 - coeff4 * yDot4;
            interpolatedDerivatives[i] =
                    coeffDot1 * yDot1 + coeffDot2 * yDot2 + coeffDot3 * yDot3 + coeffDot4 * yDot4;
        }
    }

  }

}
