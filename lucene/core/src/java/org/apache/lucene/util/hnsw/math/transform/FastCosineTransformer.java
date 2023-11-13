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


public class FastCosineTransformer implements RealTransformer, Serializable {

    
    static final long serialVersionUID = 20120212L;

    
    private final DctNormalization normalization;

    
    public FastCosineTransformer(final DctNormalization normalization) {
        this.normalization = normalization;
    }

    
    public double[] transform(final double[] f, final TransformType type)
      throws MathIllegalArgumentException {
        if (type == TransformType.FORWARD) {
            if (normalization == DctNormalization.ORTHOGONAL_DCT_I) {
                final double s = FastMath.sqrt(2.0 / (f.length - 1));
                return TransformUtils.scaleArray(fct(f), s);
            }
            return fct(f);
        }
        final double s2 = 2.0 / (f.length - 1);
        final double s1;
        if (normalization == DctNormalization.ORTHOGONAL_DCT_I) {
            s1 = FastMath.sqrt(s2);
        } else {
            s1 = s2;
        }
        return TransformUtils.scaleArray(fct(f), s1);
    }

    
    public double[] transform(final UnivariateFunction f,
        final double min, final double max, final int n,
        final TransformType type) throws MathIllegalArgumentException {

        final double[] data = FunctionUtils.sample(f, min, max, n);
        return transform(data, type);
    }

    
    protected double[] fct(double[] f)
        throws MathIllegalArgumentException {

        final double[] transformed = new double[f.length];

        final int n = f.length - 1;
        if (!ArithmeticUtils.isPowerOfTwo(n)) {
            throw new MathIllegalArgumentException(
                LocalizedFormats.NOT_POWER_OF_TWO_PLUS_ONE,
                Integer.valueOf(f.length));
        }
        if (n == 1) {       // trivial case
            transformed[0] = 0.5 * (f[0] + f[1]);
            transformed[1] = 0.5 * (f[0] - f[1]);
            return transformed;
        }

        // construct a new array and perform FFT on it
        final double[] x = new double[n];
        x[0] = 0.5 * (f[0] + f[n]);
        x[n >> 1] = f[n >> 1];
        // temporary variable for transformed[1]
        double t1 = 0.5 * (f[0] - f[n]);
        for (int i = 1; i < (n >> 1); i++) {
            final double a = 0.5 * (f[i] + f[n - i]);
            final double b = FastMath.sin(i * FastMath.PI / n) * (f[i] - f[n - i]);
            final double c = FastMath.cos(i * FastMath.PI / n) * (f[i] - f[n - i]);
            x[i] = a - b;
            x[n - i] = a + b;
            t1 += c;
        }
        FastFourierTransformer transformer;
        transformer = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] y = transformer.transform(x, TransformType.FORWARD);

        // reconstruct the FCT result for the original array
        transformed[0] = y[0].getReal();
        transformed[1] = t1;
        for (int i = 1; i < (n >> 1); i++) {
            transformed[2 * i]     = y[i].getReal();
            transformed[2 * i + 1] = transformed[2 * i - 1] - y[i].getImaginary();
        }
        transformed[n] = y[n >> 1].getReal();

        return transformed;
    }
}
