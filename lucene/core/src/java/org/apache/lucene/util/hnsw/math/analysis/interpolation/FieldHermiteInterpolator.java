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
package org.apache.lucene.util.hnsw.math.analysis.interpolation;

import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class FieldHermiteInterpolator<T extends FieldElement<T>> {

    
    private final List<T> abscissae;

    
    private final List<T[]> topDiagonal;

    
    private final List<T[]> bottomDiagonal;

    
    public FieldHermiteInterpolator() {
        this.abscissae      = new ArrayList<T>();
        this.topDiagonal    = new ArrayList<T[]>();
        this.bottomDiagonal = new ArrayList<T[]>();
    }

    
    public void addSamplePoint(final T x, final T[] ... value)
        throws ZeroException, MathArithmeticException,
               DimensionMismatchException, NullArgumentException {

        MathUtils.checkNotNull(x);
        T factorial = x.getField().getOne();
        for (int i = 0; i < value.length; ++i) {

            final T[] y = value[i].clone();
            if (i > 1) {
                factorial = factorial.multiply(i);
                final T inv = factorial.reciprocal();
                for (int j = 0; j < y.length; ++j) {
                    y[j] = y[j].multiply(inv);
                }
            }

            // update the bottom diagonal of the divided differences array
            final int n = abscissae.size();
            bottomDiagonal.add(n - i, y);
            T[] bottom0 = y;
            for (int j = i; j < n; ++j) {
                final T[] bottom1 = bottomDiagonal.get(n - (j + 1));
                if (x.equals(abscissae.get(n - (j + 1)))) {
                    throw new ZeroException(LocalizedFormats.DUPLICATED_ABSCISSA_DIVISION_BY_ZERO, x);
                }
                final T inv = x.subtract(abscissae.get(n - (j + 1))).reciprocal();
                for (int k = 0; k < y.length; ++k) {
                    bottom1[k] = inv.multiply(bottom0[k].subtract(bottom1[k]));
                }
                bottom0 = bottom1;
            }

            // update the top diagonal of the divided differences array
            topDiagonal.add(bottom0.clone());

            // update the abscissae array
            abscissae.add(x);

        }

    }

    
    public T[] value(T x) throws NoDataException, NullArgumentException {

        // safety check
        MathUtils.checkNotNull(x);
        if (abscissae.isEmpty()) {
            throw new NoDataException(LocalizedFormats.EMPTY_INTERPOLATION_SAMPLE);
        }

        final T[] value = MathArrays.buildArray(x.getField(), topDiagonal.get(0).length);
        T valueCoeff = x.getField().getOne();
        for (int i = 0; i < topDiagonal.size(); ++i) {
            T[] dividedDifference = topDiagonal.get(i);
            for (int k = 0; k < value.length; ++k) {
                value[k] = value[k].add(dividedDifference[k].multiply(valueCoeff));
            }
            final T deltaX = x.subtract(abscissae.get(i));
            valueCoeff = valueCoeff.multiply(deltaX);
        }

        return value;

    }

    
    public T[][] derivatives(T x, int order) throws NoDataException, NullArgumentException {

        // safety check
        MathUtils.checkNotNull(x);
        if (abscissae.isEmpty()) {
            throw new NoDataException(LocalizedFormats.EMPTY_INTERPOLATION_SAMPLE);
        }

        final T zero = x.getField().getZero();
        final T one  = x.getField().getOne();
        final T[] tj = MathArrays.buildArray(x.getField(), order + 1);
        tj[0] = zero;
        for (int i = 0; i < order; ++i) {
            tj[i + 1] = tj[i].add(one);
        }

        final T[][] derivatives =
                MathArrays.buildArray(x.getField(), order + 1, topDiagonal.get(0).length);
        final T[] valueCoeff = MathArrays.buildArray(x.getField(), order + 1);
        valueCoeff[0] = x.getField().getOne();
        for (int i = 0; i < topDiagonal.size(); ++i) {
            T[] dividedDifference = topDiagonal.get(i);
            final T deltaX = x.subtract(abscissae.get(i));
            for (int j = order; j >= 0; --j) {
                for (int k = 0; k < derivatives[j].length; ++k) {
                    derivatives[j][k] =
                            derivatives[j][k].add(dividedDifference[k].multiply(valueCoeff[j]));
                }
                valueCoeff[j] = valueCoeff[j].multiply(deltaX);
                if (j > 0) {
                    valueCoeff[j] = valueCoeff[j].add(tj[j].multiply(valueCoeff[j - 1]));
                }
            }
        }

        return derivatives;

    }

}
