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
import org.apache.lucene.util.hnsw.math.ode.sampling.AbstractFieldStepInterpolator;
import org.apache.lucene.util.hnsw.math.util.MathArrays;



abstract class RungeKuttaFieldStepInterpolator<T extends RealFieldElement<T>>
    extends AbstractFieldStepInterpolator<T> {

    
    private final Field<T> field;

    
    private final T[][] yDotK;

    
    protected RungeKuttaFieldStepInterpolator(final Field<T> field, final boolean forward,
                                              final T[][] yDotK,
                                              final FieldODEStateAndDerivative<T> globalPreviousState,
                                              final FieldODEStateAndDerivative<T> globalCurrentState,
                                              final FieldODEStateAndDerivative<T> softPreviousState,
                                              final FieldODEStateAndDerivative<T> softCurrentState,
                                              final FieldEquationsMapper<T> mapper) {
        super(forward, globalPreviousState, globalCurrentState, softPreviousState, softCurrentState, mapper);
        this.field = field;
        this.yDotK = MathArrays.buildArray(field, yDotK.length, -1);
        for (int i = 0; i < yDotK.length; ++i) {
            this.yDotK[i] = yDotK[i].clone();
        }
    }

    
    @Override
    protected RungeKuttaFieldStepInterpolator<T> create(boolean newForward,
                                                        FieldODEStateAndDerivative<T> newGlobalPreviousState,
                                                        FieldODEStateAndDerivative<T> newGlobalCurrentState,
                                                        FieldODEStateAndDerivative<T> newSoftPreviousState,
                                                        FieldODEStateAndDerivative<T> newSoftCurrentState,
                                                        FieldEquationsMapper<T> newMapper) {
        return create(field, newForward, yDotK,
                      newGlobalPreviousState, newGlobalCurrentState,
                      newSoftPreviousState, newSoftCurrentState,
                      newMapper);
    }

    
    protected abstract RungeKuttaFieldStepInterpolator<T> create(Field<T> newField, boolean newForward, T[][] newYDotK,
                                                                 FieldODEStateAndDerivative<T> newGlobalPreviousState,
                                                                 FieldODEStateAndDerivative<T> newGlobalCurrentState,
                                                                 FieldODEStateAndDerivative<T> newSoftPreviousState,
                                                                 FieldODEStateAndDerivative<T> newSoftCurrentState,
                                                                 FieldEquationsMapper<T> newMapper);

    
    protected final T[] previousStateLinearCombination(final T ... coefficients) {
        return combine(getPreviousState().getState(),
                       coefficients);
    }

    
    protected T[] currentStateLinearCombination(final T ... coefficients) {
        return combine(getCurrentState().getState(),
                       coefficients);
    }

    
    protected T[] derivativeLinearCombination(final T ... coefficients) {
        return combine(MathArrays.buildArray(field, yDotK[0].length), coefficients);
    }

    
    private T[] combine(final T[] a, final T ... coefficients) {
        for (int i = 0; i < a.length; ++i) {
            for (int k = 0; k < coefficients.length; ++k) {
                a[i] = a[i].add(coefficients[k].multiply(yDotK[k][i]));
            }
        }
        return a;
    }

}
