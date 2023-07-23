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

import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.random.UnitSphereRandomVectorGenerator;


@Deprecated
public class MicrosphereInterpolator
    implements MultivariateInterpolator {
    
    public static final int DEFAULT_MICROSPHERE_ELEMENTS = 2000;
    
    public static final int DEFAULT_BRIGHTNESS_EXPONENT = 2;
    
    private final int microsphereElements;
    
    private final int brightnessExponent;

    
    public MicrosphereInterpolator() {
        this(DEFAULT_MICROSPHERE_ELEMENTS, DEFAULT_BRIGHTNESS_EXPONENT);
    }

    
    public MicrosphereInterpolator(final int elements,
                                   final int exponent)
        throws NotPositiveException,
               NotStrictlyPositiveException {
        if (exponent < 0) {
            throw new NotPositiveException(exponent);
        }
        if (elements <= 0) {
            throw new NotStrictlyPositiveException(elements);
        }

        microsphereElements = elements;
        brightnessExponent = exponent;
    }

    
    public MultivariateFunction interpolate(final double[][] xval,
                                            final double[] yval)
        throws DimensionMismatchException,
               NoDataException,
               NullArgumentException {
        final UnitSphereRandomVectorGenerator rand
            = new UnitSphereRandomVectorGenerator(xval[0].length);
        return new MicrosphereInterpolatingFunction(xval, yval,
                                                    brightnessExponent,
                                                    microsphereElements,
                                                    rand);
    }
}
