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

package org.apache.lucene.util.hnsw.math.ode;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class FieldEquationsMapper<T extends RealFieldElement<T>> implements Serializable {

    
    private static final long serialVersionUID = 20151114L;

    
    private final int[] start;

    
    FieldEquationsMapper(final FieldEquationsMapper<T> mapper, final int dimension) {
        final int index = (mapper == null) ? 0 : mapper.getNumberOfEquations();
        this.start = new int[index + 2];
        if (mapper == null) {
            start[0] = 0;
        } else {
            System.arraycopy(mapper.start, 0, start, 0, index + 1);
        }
        start[index + 1] = start[index] + dimension;
    }

    
    public int getNumberOfEquations() {
        return start.length - 1;
    }

    
    public int getTotalDimension() {
        return start[start.length - 1];
    }

    
    public T[] mapState(final FieldODEState<T> state) {
        final T[] y = MathArrays.buildArray(state.getTime().getField(), getTotalDimension());
        int index = 0;
        insertEquationData(index, state.getState(), y);
        while (++index < getNumberOfEquations()) {
            insertEquationData(index, state.getSecondaryState(index), y);
        }
        return y;
    }

    
    public T[] mapDerivative(final FieldODEStateAndDerivative<T> state) {
        final T[] yDot = MathArrays.buildArray(state.getTime().getField(), getTotalDimension());
        int index = 0;
        insertEquationData(index, state.getDerivative(), yDot);
        while (++index < getNumberOfEquations()) {
            insertEquationData(index, state.getSecondaryDerivative(index), yDot);
        }
        return yDot;
    }

    
    public FieldODEStateAndDerivative<T> mapStateAndDerivative(final T t, final T[] y, final T[] yDot)
        throws DimensionMismatchException {

        if (y.length != getTotalDimension()) {
            throw new DimensionMismatchException(y.length, getTotalDimension());
        }

        if (yDot.length != getTotalDimension()) {
            throw new DimensionMismatchException(yDot.length, getTotalDimension());
        }

        final int n = getNumberOfEquations();
        int index = 0;
        final T[] state      = extractEquationData(index, y);
        final T[] derivative = extractEquationData(index, yDot);
        if (n < 2) {
            return new FieldODEStateAndDerivative<T>(t, state, derivative);
        } else {
            final T[][] secondaryState      = MathArrays.buildArray(t.getField(), n - 1, -1);
            final T[][] secondaryDerivative = MathArrays.buildArray(t.getField(), n - 1, -1);
            while (++index < getNumberOfEquations()) {
                secondaryState[index - 1]      = extractEquationData(index, y);
                secondaryDerivative[index - 1] = extractEquationData(index, yDot);
            }
            return new FieldODEStateAndDerivative<T>(t, state, derivative, secondaryState, secondaryDerivative);
        }
    }

    
    public T[] extractEquationData(final int index, final T[] complete)
        throws MathIllegalArgumentException, DimensionMismatchException {
        checkIndex(index);
        final int begin     = start[index];
        final int end       = start[index + 1];
        if (complete.length < end) {
            throw new DimensionMismatchException(complete.length, end);
        }
        final int dimension = end - begin;
        final T[] equationData = MathArrays.buildArray(complete[0].getField(), dimension);
        System.arraycopy(complete, begin, equationData, 0, dimension);
        return equationData;
    }

    
    public void insertEquationData(final int index, T[] equationData, T[] complete)
        throws DimensionMismatchException {
        checkIndex(index);
        final int begin     = start[index];
        final int end       = start[index + 1];
        final int dimension = end - begin;
        if (complete.length < end) {
            throw new DimensionMismatchException(complete.length, end);
        }
        if (equationData.length != dimension) {
            throw new DimensionMismatchException(equationData.length, dimension);
        }
        System.arraycopy(equationData, 0, complete, begin, dimension);
    }

    
    private void checkIndex(final int index) throws MathIllegalArgumentException {
        if (index < 0 || index > start.length - 2) {
            throw new MathIllegalArgumentException(LocalizedFormats.ARGUMENT_OUTSIDE_DOMAIN,
                                                   index, 0, start.length - 2);
        }
    }

}
