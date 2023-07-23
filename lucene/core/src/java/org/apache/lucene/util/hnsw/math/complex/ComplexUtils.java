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

package org.apache.lucene.util.hnsw.math.complex;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class ComplexUtils {

    
    private ComplexUtils() {}

    
    public static Complex polar2Complex(double r, double theta) throws MathIllegalArgumentException {
        if (r < 0) {
            throw new MathIllegalArgumentException(
                  LocalizedFormats.NEGATIVE_COMPLEX_MODULE, r);
        }
        return new Complex(r * FastMath.cos(theta), r * FastMath.sin(theta));
    }

    
    public static Complex[] convertToComplex(double[] real) {
        final Complex c[] = new Complex[real.length];
        for (int i = 0; i < real.length; i++) {
            c[i] = new Complex(real[i], 0);
        }

        return c;
    }
}
