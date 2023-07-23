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

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowFieldMatrix;
import org.apache.lucene.util.hnsw.math.ode.FieldEquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;
import org.apache.lucene.util.hnsw.math.ode.sampling.AbstractFieldStepInterpolator;
import org.apache.lucene.util.hnsw.math.util.MathArrays;



class AdamsFieldStepInterpolator<T extends RealFieldElement<T>> extends AbstractFieldStepInterpolator<T> {

    
    private T scalingH;

    
    private final FieldODEStateAndDerivative<T> reference;

    
    private final T[] scaled;

    
    private final Array2DRowFieldMatrix<T> nordsieck;

    
    AdamsFieldStepInterpolator(final T stepSize, final FieldODEStateAndDerivative<T> reference,
                               final T[] scaled, final Array2DRowFieldMatrix<T> nordsieck,
                               final boolean isForward,
                               final FieldODEStateAndDerivative<T> globalPreviousState,
                               final FieldODEStateAndDerivative<T> globalCurrentState,
                               final FieldEquationsMapper<T> equationsMapper) {
        this(stepSize, reference, scaled, nordsieck,
             isForward, globalPreviousState, globalCurrentState,
             globalPreviousState, globalCurrentState, equationsMapper);
    }

    
    private AdamsFieldStepInterpolator(final T stepSize, final FieldODEStateAndDerivative<T> reference,
                                       final T[] scaled, final Array2DRowFieldMatrix<T> nordsieck,
                                       final boolean isForward,
                                       final FieldODEStateAndDerivative<T> globalPreviousState,
                                       final FieldODEStateAndDerivative<T> globalCurrentState,
                                       final FieldODEStateAndDerivative<T> softPreviousState,
                                       final FieldODEStateAndDerivative<T> softCurrentState,
                                       final FieldEquationsMapper<T> equationsMapper) {
        super(isForward, globalPreviousState, globalCurrentState,
              softPreviousState, softCurrentState, equationsMapper);
        this.scalingH  = stepSize;
        this.reference = reference;
        this.scaled    = scaled.clone();
        this.nordsieck = new Array2DRowFieldMatrix<T>(nordsieck.getData(), false);
    }

    
    @Override
    protected AdamsFieldStepInterpolator<T> create(boolean newForward,
                                                   FieldODEStateAndDerivative<T> newGlobalPreviousState,
                                                   FieldODEStateAndDerivative<T> newGlobalCurrentState,
                                                   FieldODEStateAndDerivative<T> newSoftPreviousState,
                                                   FieldODEStateAndDerivative<T> newSoftCurrentState,
                                                   FieldEquationsMapper<T> newMapper) {
        return new AdamsFieldStepInterpolator<T>(scalingH, reference, scaled, nordsieck,
                                                 newForward,
                                                 newGlobalPreviousState, newGlobalCurrentState,
                                                 newSoftPreviousState, newSoftCurrentState,
                                                 newMapper);

    }

    
    @Override
    protected FieldODEStateAndDerivative<T> computeInterpolatedStateAndDerivatives(final FieldEquationsMapper<T> equationsMapper,
                                                                                   final T time, final T theta,
                                                                                   final T thetaH, final T oneMinusThetaH) {
        return taylor(reference, time, scalingH, scaled, nordsieck);
    }

    
    public static <S extends RealFieldElement<S>> FieldODEStateAndDerivative<S> taylor(final FieldODEStateAndDerivative<S> reference,
                                                                                       final S time, final S stepSize,
                                                                                       final S[] scaled,
                                                                                       final Array2DRowFieldMatrix<S> nordsieck) {

        final S x = time.subtract(reference.getTime());
        final S normalizedAbscissa = x.divide(stepSize);

        S[] stateVariation = MathArrays.buildArray(time.getField(), scaled.length);
        Arrays.fill(stateVariation, time.getField().getZero());
        S[] estimatedDerivatives = MathArrays.buildArray(time.getField(), scaled.length);
        Arrays.fill(estimatedDerivatives, time.getField().getZero());

        // apply Taylor formula from high order to low order,
        // for the sake of numerical accuracy
        final S[][] nData = nordsieck.getDataRef();
        for (int i = nData.length - 1; i >= 0; --i) {
            final int order = i + 2;
            final S[] nDataI = nData[i];
            final S power = normalizedAbscissa.pow(order);
            for (int j = 0; j < nDataI.length; ++j) {
                final S d = nDataI[j].multiply(power);
                stateVariation[j]          = stateVariation[j].add(d);
                estimatedDerivatives[j] = estimatedDerivatives[j].add(d.multiply(order));
            }
        }

        S[] estimatedState = reference.getState();
        for (int j = 0; j < stateVariation.length; ++j) {
            stateVariation[j]    = stateVariation[j].add(scaled[j].multiply(normalizedAbscissa));
            estimatedState[j] = estimatedState[j].add(stateVariation[j]);
            estimatedDerivatives[j] =
                estimatedDerivatives[j].add(scaled[j].multiply(normalizedAbscissa)).divide(x);
        }

        return new FieldODEStateAndDerivative<S>(time, estimatedState, estimatedDerivatives);

    }

}
