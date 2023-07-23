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
import org.apache.lucene.util.hnsw.math.complex.Complex;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.ArithmeticUtils;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class FastSineTransformer implements RealTransformer, Serializable {

    
    static final long serialVersionUID = 20120211L;

    
    private final DstNormalization normalization;

    
    public FastSineTransformer(final DstNormalization normalization) {
        this.normalization = normalization;
    }

    
    public double[] transform(final double[] f, final TransformType type) {
        if (normalization == DstNormalization.ORTHOGONAL_DST_I) {
            final double s = FastMath.sqrt(2.0 / f.length);
            return TransformUtils.scaleArray(fst(f), s);
        }
        if (type == TransformType.FORWARD) {
            return fst(f);
        }
        final double s = 2.0 / f.length;
        return TransformUtils.scaleArray(fst(f), s);
    }

    
    public double[] transform(final UnivariateFunction f,
        final double min, final double max, final int n,
        final TransformType type) {

        final double[] data = FunctionUtils.sample(f, min, max, n);
        data[0] = 0.0;
        return transform(data, type);
    }

    
    protected double[] fst(double[] f) throws MathIllegalArgumentException {

        final double[] transformed = new double[f.length];

        if (!ArithmeticUtils.isPowerOfTwo(f.length)) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.NOT_POWER_OF_TWO_CONSIDER_PADDING,
                    Integer.valueOf(f.length));
        }
        if (f[0] != 0.0) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.FIRST_ELEMENT_NOT_ZERO,
                    Double.valueOf(f[0]));
        }
        final int n = f.length;
        if (n == 1) {       // trivial case
            transformed[0] = 0.0;
            return transformed;
        }

        // construct a new array and perform FFT on it
        final double[] x = new double[n];
        x[0] = 0.0;
        x[n >> 1] = 2.0 * f[n >> 1];
        for (int i = 1; i < (n >> 1); i++) {
            final double a = FastMath.sin(i * FastMath.PI / n) * (f[i] + f[n - i]);
            final double b = 0.5 * (f[i] - f[n - i]);
            x[i]     = a + b;
            x[n - i] = a - b;
        }
        FastFourierTransformer transformer;
        transformer = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] y = transformer.transform(x, TransformType.FORWARD);

        // reconstruct the FST result for the original array
        transformed[0] = 0.0;
        transformed[1] = 0.5 * y[0].getReal();
        for (int i = 1; i < (n >> 1); i++) {
            transformed[2 * i]     = -y[i].getImaginary();
            transformed[2 * i + 1] = y[i].getReal() + transformed[2 * i - 1];
        }

        return transformed;
    }
}
