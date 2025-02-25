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
import org.apache.lucene.util.hnsw.math.util.MathUtils;




public class HighamHall54FieldIntegrator<T extends RealFieldElement<T>>
    extends EmbeddedRungeKuttaFieldIntegrator<T> {

    
    private static final String METHOD_NAME = "Higham-Hall 5(4)";

    
    private final T[] e ;

    
    public HighamHall54FieldIntegrator(final Field<T> field,
                                       final double minStep, final double maxStep,
                                       final double scalAbsoluteTolerance,
                                       final double scalRelativeTolerance) {
        super(field, METHOD_NAME, -1,
              minStep, maxStep, scalAbsoluteTolerance, scalRelativeTolerance);
        e = MathArrays.buildArray(field, 7);
        e[0] = fraction(-1,  20);
        e[1] = field.getZero();
        e[2] = fraction(81, 160);
        e[3] = fraction(-6,   5);
        e[4] = fraction(25,  32);
        e[5] = fraction( 1,  16);
        e[6] = fraction(-1,  10);
    }

    
    public HighamHall54FieldIntegrator(final Field<T> field,
                                       final double minStep, final double maxStep,
                                       final double[] vecAbsoluteTolerance,
                                       final double[] vecRelativeTolerance) {
        super(field, METHOD_NAME, -1,
              minStep, maxStep, vecAbsoluteTolerance, vecRelativeTolerance);
        e = MathArrays.buildArray(field, 7);
        e[0] = fraction(-1,  20);
        e[1] = field.getZero();
        e[2] = fraction(81, 160);
        e[3] = fraction(-6,   5);
        e[4] = fraction(25,  32);
        e[5] = fraction( 1,  16);
        e[6] = fraction(-1,  10);
    }

    
    public T[] getC() {
        final T[] c = MathArrays.buildArray(getField(), 6);
        c[0] = fraction(2, 9);
        c[1] = fraction(1, 3);
        c[2] = fraction(1, 2);
        c[3] = fraction(3, 5);
        c[4] = getField().getOne();
        c[5] = getField().getOne();
        return c;
    }

    
    public T[][] getA() {
        final T[][] a = MathArrays.buildArray(getField(), 6, -1);
        for (int i = 0; i < a.length; ++i) {
            a[i] = MathArrays.buildArray(getField(), i + 1);
        }
        a[0][0] = fraction(     2,     9);
        a[1][0] = fraction(     1,    12);
        a[1][1] = fraction(     1,     4);
        a[2][0] = fraction(     1,     8);
        a[2][1] = getField().getZero();
        a[2][2] = fraction(     3,     8);
        a[3][0] = fraction(    91,   500);
        a[3][1] = fraction(   -27,   100);
        a[3][2] = fraction(    78,   125);
        a[3][3] = fraction(     8,   125);
        a[4][0] = fraction(   -11,    20);
        a[4][1] = fraction(    27,    20);
        a[4][2] = fraction(    12,     5);
        a[4][3] = fraction(   -36,     5);
        a[4][4] = fraction(     5,     1);
        a[5][0] = fraction(     1,    12);
        a[5][1] = getField().getZero();
        a[5][2] = fraction(    27,    32);
        a[5][3] = fraction(    -4,     3);
        a[5][4] = fraction(   125,    96);
        a[5][5] = fraction(     5,    48);
        return a;
    }

    
    public T[] getB() {
        final T[] b = MathArrays.buildArray(getField(), 7);
        b[0] = fraction(  1, 12);
        b[1] = getField().getZero();
        b[2] = fraction( 27, 32);
        b[3] = fraction( -4,  3);
        b[4] = fraction(125, 96);
        b[5] = fraction(  5, 48);
        b[6] = getField().getZero();
        return b;
    }

    
    @Override
    protected HighamHall54FieldStepInterpolator<T>
        createInterpolator(final boolean forward, T[][] yDotK,
                           final FieldODEStateAndDerivative<T> globalPreviousState,
                           final FieldODEStateAndDerivative<T> globalCurrentState, final FieldEquationsMapper<T> mapper) {
        return new HighamHall54FieldStepInterpolator<T>(getField(), forward, yDotK,
                                                        globalPreviousState, globalCurrentState,
                                                        globalPreviousState, globalCurrentState,
                                                        mapper);
    }

    
    @Override
    public int getOrder() {
        return 5;
    }

    
    @Override
    protected T estimateError(final T[][] yDotK, final T[] y0, final T[] y1, final T h) {

        T error = getField().getZero();

        for (int j = 0; j < mainSetDimension; ++j) {
            T errSum = yDotK[0][j].multiply(e[0]);
            for (int l = 1; l < e.length; ++l) {
                errSum = errSum.add(yDotK[l][j].multiply(e[l]));
            }

            final T yScale = MathUtils.max(y0[j].abs(), y1[j].abs());
            final T tol    = (vecAbsoluteTolerance == null) ?
                             yScale.multiply(scalRelativeTolerance).add(scalAbsoluteTolerance) :
                             yScale.multiply(vecRelativeTolerance[j]).add(vecAbsoluteTolerance[j]);
            final T ratio  = h.multiply(errSum).divide(tol);
            error = error.add(ratio.multiply(ratio));

        }

        return error.divide(mainSetDimension).sqrt();

    }

}
