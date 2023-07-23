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
import org.apache.lucene.util.hnsw.math.ode.FieldEquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;



class EulerFieldStepInterpolator<T extends RealFieldElement<T>>
    extends RungeKuttaFieldStepInterpolator<T> {

    
    EulerFieldStepInterpolator(final Field<T> field, final boolean forward,
                                             final T[][] yDotK,
                                             final FieldODEStateAndDerivative<T> globalPreviousState,
                                             final FieldODEStateAndDerivative<T> globalCurrentState,
                                             final FieldODEStateAndDerivative<T> softPreviousState,
                                             final FieldODEStateAndDerivative<T> softCurrentState,
                                             final FieldEquationsMapper<T> mapper) {
        super(field, forward, yDotK,
              globalPreviousState, globalCurrentState, softPreviousState, softCurrentState,
              mapper);
    }

    
    @Override
    protected EulerFieldStepInterpolator<T> create(final Field<T> newField, final boolean newForward, final T[][] newYDotK,
                                                                 final FieldODEStateAndDerivative<T> newGlobalPreviousState,
                                                                 final FieldODEStateAndDerivative<T> newGlobalCurrentState,
                                                                 final FieldODEStateAndDerivative<T> newSoftPreviousState,
                                                                 final FieldODEStateAndDerivative<T> newSoftCurrentState,
                                                                 final FieldEquationsMapper<T> newMapper) {
        return new EulerFieldStepInterpolator<T>(newField, newForward, newYDotK,
                                                 newGlobalPreviousState, newGlobalCurrentState,
                                                 newSoftPreviousState, newSoftCurrentState,
                                                 newMapper);
    }

    
    @SuppressWarnings("unchecked")
    @Override
    protected FieldODEStateAndDerivative<T> computeInterpolatedStateAndDerivatives(final FieldEquationsMapper<T> mapper,
                                                                                   final T time, final T theta,
                                                                                   final T thetaH, final T oneMinusThetaH) {
        final T[] interpolatedState;
        final T[] interpolatedDerivatives;
        if (getGlobalPreviousState() != null && theta.getReal() <= 0.5) {
            interpolatedState       = previousStateLinearCombination(thetaH);
            interpolatedDerivatives = derivativeLinearCombination(time.getField().getOne());
        } else {
            interpolatedState       = currentStateLinearCombination(oneMinusThetaH.negate());
            interpolatedDerivatives = derivativeLinearCombination(time.getField().getOne());
        }

        return new FieldODEStateAndDerivative<T>(time, interpolatedState, interpolatedDerivatives);

    }

}
