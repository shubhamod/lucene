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



public class ThreeEighthesFieldIntegrator<T extends RealFieldElement<T>>
    extends RungeKuttaFieldIntegrator<T> {

    
    public ThreeEighthesFieldIntegrator(final Field<T> field, final T step) {
        super(field, "3/8", step);
    }

    
    public T[] getC() {
        final T[] c = MathArrays.buildArray(getField(), 3);
        c[0] = fraction(1, 3);
        c[1] = c[0].add(c[0]);
        c[2] = getField().getOne();
        return c;
    }

    
    public T[][] getA() {
        final T[][] a = MathArrays.buildArray(getField(), 3, -1);
        for (int i = 0; i < a.length; ++i) {
            a[i] = MathArrays.buildArray(getField(), i + 1);
        }
        a[0][0] = fraction(1, 3);
        a[1][0] = a[0][0].negate();
        a[1][1] = getField().getOne();
        a[2][0] = getField().getOne();
        a[2][1] = getField().getOne().negate();
        a[2][2] = getField().getOne();
        return a;
    }

    
    public T[] getB() {
        final T[] b = MathArrays.buildArray(getField(), 4);
        b[0] = fraction(1, 8);
        b[1] = fraction(3, 8);
        b[2] = b[1];
        b[3] = b[0];
        return b;
    }

    
    @Override
    protected ThreeEighthesFieldStepInterpolator<T>
        createInterpolator(final boolean forward, T[][] yDotK,
                           final FieldODEStateAndDerivative<T> globalPreviousState,
                           final FieldODEStateAndDerivative<T> globalCurrentState,
                           final FieldEquationsMapper<T> mapper) {
        return new ThreeEighthesFieldStepInterpolator<T>(getField(), forward, yDotK,
                                                         globalPreviousState, globalCurrentState,
                                                         globalPreviousState, globalCurrentState,
                                                         mapper);
    }

}
