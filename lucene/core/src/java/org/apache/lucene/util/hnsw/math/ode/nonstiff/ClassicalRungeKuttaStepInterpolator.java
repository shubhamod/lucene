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



class ClassicalRungeKuttaStepInterpolator
    extends RungeKuttaStepInterpolator {

    
    private static final long serialVersionUID = 20111120L;

    
    // CHECKSTYLE: stop RedundantModifier
    // the public modifier here is needed for serialization
    public ClassicalRungeKuttaStepInterpolator() {
    }
    // CHECKSTYLE: resume RedundantModifier

    
    ClassicalRungeKuttaStepInterpolator(final ClassicalRungeKuttaStepInterpolator interpolator) {
        super(interpolator);
    }

    
    @Override
    protected StepInterpolator doCopy() {
        return new ClassicalRungeKuttaStepInterpolator(this);
    }

    
    @Override
    protected void computeInterpolatedStateAndDerivatives(final double theta,
                                            final double oneMinusThetaH) {

        final double oneMinusTheta  = 1 - theta;
        final double oneMinus2Theta = 1 - 2 * theta;
        final double coeffDot1     = oneMinusTheta * oneMinus2Theta;
        final double coeffDot23    = 2 * theta * oneMinusTheta;
        final double coeffDot4     = -theta * oneMinus2Theta;
        if ((previousState != null) && (theta <= 0.5)) {
            final double fourTheta2     = 4 * theta * theta;
            final double s             = theta * h / 6.0;
            final double coeff1        = s * ( 6 - 9 * theta + fourTheta2);
            final double coeff23       = s * ( 6 * theta - fourTheta2);
            final double coeff4        = s * (-3 * theta + fourTheta2);
            for (int i = 0; i < interpolatedState.length; ++i) {
                final double yDot1  = yDotK[0][i];
                final double yDot23 = yDotK[1][i] + yDotK[2][i];
                final double yDot4  = yDotK[3][i];
                interpolatedState[i] =
                        previousState[i] + coeff1  * yDot1 + coeff23 * yDot23 + coeff4  * yDot4;
                interpolatedDerivatives[i] =
                        coeffDot1 * yDot1 + coeffDot23 * yDot23 + coeffDot4 * yDot4;
            }
        } else {
            final double fourTheta      = 4 * theta;
            final double s             = oneMinusThetaH / 6.0;
            final double coeff1        = s * ((-fourTheta + 5) * theta - 1);
            final double coeff23       = s * (( fourTheta - 2) * theta - 2);
            final double coeff4        = s * ((-fourTheta - 1) * theta - 1);
            for (int i = 0; i < interpolatedState.length; ++i) {
                final double yDot1  = yDotK[0][i];
                final double yDot23 = yDotK[1][i] + yDotK[2][i];
                final double yDot4  = yDotK[3][i];
                interpolatedState[i] =
                        currentState[i] + coeff1  * yDot1 + coeff23 * yDot23 + coeff4  * yDot4;
                interpolatedDerivatives[i] =
                        coeffDot1 * yDot1 + coeffDot23 * yDot23 + coeffDot4 * yDot4;
            }
        }

    }

}
