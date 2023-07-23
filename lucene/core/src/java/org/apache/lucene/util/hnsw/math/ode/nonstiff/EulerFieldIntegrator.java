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
import org.apache.lucene.util.hnsw.math.util.MathArrays;



public class EulerFieldIntegrator<T extends RealFieldElement<T>> extends RungeKuttaFieldIntegrator<T> {

    
    public EulerFieldIntegrator(final Field<T> field, final T step) {
        super(field, "Euler", step);
    }

    
    public T[] getC() {
        return MathArrays.buildArray(getField(), 0);
    }

    
    public T[][] getA() {
        return MathArrays.buildArray(getField(), 0, 0);
    }

    
    public T[] getB() {
        final T[] b = MathArrays.buildArray(getField(), 1);
        b[0] = getField().getOne();
        return b;
    }

    
    @Override
    protected EulerFieldStepInterpolator<T>
        createInterpolator(final boolean forward, T[][] yDotK,
                           final FieldODEStateAndDerivative<T> globalPreviousState,
                           final FieldODEStateAndDerivative<T> globalCurrentState,
                           final FieldEquationsMapper<T> mapper) {
        return new EulerFieldStepInterpolator<T>(getField(), forward, yDotK,
                                                 globalPreviousState, globalCurrentState,
                                                 globalPreviousState, globalCurrentState,
                                                 mapper);
    }

}
