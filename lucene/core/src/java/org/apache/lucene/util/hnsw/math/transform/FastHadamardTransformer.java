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
package org.apache.lucene.util.hnsw.math.transform;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.analysis.FunctionUtils;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.ArithmeticUtils;


public class FastHadamardTransformer implements RealTransformer, Serializable {

    
    static final long serialVersionUID = 20120211L;

    
    public double[] transform(final double[] f, final TransformType type) {
        if (type == TransformType.FORWARD) {
            return fht(f);
        }
        return TransformUtils.scaleArray(fht(f), 1.0 / f.length);
    }

    
    public double[] transform(final UnivariateFunction f,
        final double min, final double max, final int n,
        final TransformType type) {

        return transform(FunctionUtils.sample(f, min, max, n), type);
    }

    
    public int[] transform(final int[] f) {
        return fht(f);
    }

    
    protected double[] fht(double[] x) throws MathIllegalArgumentException {

        final int n     = x.length;
        final int halfN = n / 2;

        if (!ArithmeticUtils.isPowerOfTwo(n)) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.NOT_POWER_OF_TWO,
                    Integer.valueOf(n));
        }

        /*
         * Instead of creating a matrix with p+1 columns and n rows, we use two
         * one dimension arrays which we are used in an alternating way.
         */
        double[] yPrevious = new double[n];
        double[] yCurrent  = x.clone();

        // iterate from left to right (column)
        for (int j = 1; j < n; j <<= 1) {

            // switch columns
            final double[] yTmp = yCurrent;
            yCurrent  = yPrevious;
            yPrevious = yTmp;

            // iterate from top to bottom (row)
            for (int i = 0; i < halfN; ++i) {
                // Dtop: the top part works with addition
                final int twoI = 2 * i;
                yCurrent[i] = yPrevious[twoI] + yPrevious[twoI + 1];
            }
            for (int i = halfN; i < n; ++i) {
                // Dbottom: the bottom part works with subtraction
                final int twoI = 2 * i;
                yCurrent[i] = yPrevious[twoI - n] - yPrevious[twoI - n + 1];
            }
        }

        return yCurrent;

    }

    
    protected int[] fht(int[] x) throws MathIllegalArgumentException {

        final int n     = x.length;
        final int halfN = n / 2;

        if (!ArithmeticUtils.isPowerOfTwo(n)) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.NOT_POWER_OF_TWO,
                    Integer.valueOf(n));
        }

        /*
         * Instead of creating a matrix with p+1 columns and n rows, we use two
         * one dimension arrays which we are used in an alternating way.
         */
        int[] yPrevious = new int[n];
        int[] yCurrent  = x.clone();

        // iterate from left to right (column)
        for (int j = 1; j < n; j <<= 1) {

            // switch columns
            final int[] yTmp = yCurrent;
            yCurrent  = yPrevious;
            yPrevious = yTmp;

            // iterate from top to bottom (row)
            for (int i = 0; i < halfN; ++i) {
                // Dtop: the top part works with addition
                final int twoI = 2 * i;
                yCurrent[i] = yPrevious[twoI] + yPrevious[twoI + 1];
            }
            for (int i = halfN; i < n; ++i) {
                // Dbottom: the bottom part works with subtraction
                final int twoI = 2 * i;
                yCurrent[i] = yPrevious[twoI - n] - yPrevious[twoI - n + 1];
            }
        }

        // return the last computed output vector y
        return yCurrent;

    }

}
