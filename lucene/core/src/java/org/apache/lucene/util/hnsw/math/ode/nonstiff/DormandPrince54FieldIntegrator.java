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




public class DormandPrince54FieldIntegrator<T extends RealFieldElement<T>>
    extends EmbeddedRungeKuttaFieldIntegrator<T> {

    
    private static final String METHOD_NAME = "Dormand-Prince 5(4)";

    
    private final T e1;

    // element 2 is zero, so it is neither stored nor used

    
    private final T e3;

    
    private final T e4;

    
    private final T e5;

    
    private final T e6;

    
    private final T e7;

    
    public DormandPrince54FieldIntegrator(final Field<T> field,
                                          final double minStep, final double maxStep,
                                          final double scalAbsoluteTolerance,
                                          final double scalRelativeTolerance) {
        super(field, METHOD_NAME, 6,
              minStep, maxStep, scalAbsoluteTolerance, scalRelativeTolerance);
        e1 = fraction(    71,  57600);
        e3 = fraction(   -71,  16695);
        e4 = fraction(    71,   1920);
        e5 = fraction(-17253, 339200);
        e6 = fraction(    22,    525);
        e7 = fraction(    -1,     40);
    }

    
    public DormandPrince54FieldIntegrator(final Field<T> field,
                                          final double minStep, final double maxStep,
                                          final double[] vecAbsoluteTolerance,
                                          final double[] vecRelativeTolerance) {
        super(field, METHOD_NAME, 6,
              minStep, maxStep, vecAbsoluteTolerance, vecRelativeTolerance);
        e1 = fraction(    71,  57600);
        e3 = fraction(   -71,  16695);
        e4 = fraction(    71,   1920);
        e5 = fraction(-17253, 339200);
        e6 = fraction(    22,    525);
        e7 = fraction(    -1,     40);
    }

    
    public T[] getC() {
        final T[] c = MathArrays.buildArray(getField(), 6);
        c[0] = fraction(1,  5);
        c[1] = fraction(3, 10);
        c[2] = fraction(4,  5);
        c[3] = fraction(8,  9);
        c[4] = getField().getOne();
        c[5] = getField().getOne();
        return c;
    }

    
    public T[][] getA() {
        final T[][] a = MathArrays.buildArray(getField(), 6, -1);
        for (int i = 0; i < a.length; ++i) {
            a[i] = MathArrays.buildArray(getField(), i + 1);
        }
        a[0][0] = fraction(     1,     5);
        a[1][0] = fraction(     3,    40);
        a[1][1] = fraction(     9,    40);
        a[2][0] = fraction(    44,    45);
        a[2][1] = fraction(   -56,    15);
        a[2][2] = fraction(    32,     9);
        a[3][0] = fraction( 19372,  6561);
        a[3][1] = fraction(-25360,  2187);
        a[3][2] = fraction( 64448,  6561);
        a[3][3] = fraction(  -212,   729);
        a[4][0] = fraction(  9017,  3168);
        a[4][1] = fraction(  -355,    33);
        a[4][2] = fraction( 46732,  5247);
        a[4][3] = fraction(    49,   176);
        a[4][4] = fraction( -5103, 18656);
        a[5][0] = fraction(    35,   384);
        a[5][1] = getField().getZero();
        a[5][2] = fraction(   500,  1113);
        a[5][3] = fraction(   125,   192);
        a[5][4] = fraction( -2187,  6784);
        a[5][5] = fraction(    11,    84);
        return a;
    }

    
    public T[] getB() {
        final T[] b = MathArrays.buildArray(getField(), 7);
        b[0] = fraction(   35,   384);
        b[1] = getField().getZero();
        b[2] = fraction(  500, 1113);
        b[3] = fraction(  125,  192);
        b[4] = fraction(-2187, 6784);
        b[5] = fraction(   11,   84);
        b[6] = getField().getZero();
        return b;
    }

    
    @Override
    protected DormandPrince54FieldStepInterpolator<T>
        createInterpolator(final boolean forward, T[][] yDotK,
                           final FieldODEStateAndDerivative<T> globalPreviousState,
                           final FieldODEStateAndDerivative<T> globalCurrentState, final FieldEquationsMapper<T> mapper) {
        return new DormandPrince54FieldStepInterpolator<T>(getField(), forward, yDotK,
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
            final T errSum =     yDotK[0][j].multiply(e1).
                             add(yDotK[2][j].multiply(e3)).
                             add(yDotK[3][j].multiply(e4)).
                             add(yDotK[4][j].multiply(e5)).
                             add(yDotK[5][j].multiply(e6)).
                             add(yDotK[6][j].multiply(e7));

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
