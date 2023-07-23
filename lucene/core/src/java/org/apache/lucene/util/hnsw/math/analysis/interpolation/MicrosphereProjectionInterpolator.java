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
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.random.UnitSphereRandomVectorGenerator;


public class MicrosphereProjectionInterpolator
    implements MultivariateInterpolator {
    
    private final double exponent;
    
    private final InterpolatingMicrosphere microsphere;
    
    private final boolean sharedSphere;
    
    private final double noInterpolationTolerance;

    
    public MicrosphereProjectionInterpolator(int dimension,
                                             int elements,
                                             double maxDarkFraction,
                                             double darkThreshold,
                                             double background,
                                             double exponent,
                                             boolean sharedSphere,
                                             double noInterpolationTolerance) {
        this(new InterpolatingMicrosphere(dimension,
                                          elements,
                                          maxDarkFraction,
                                          darkThreshold,
                                          background,
                                          new UnitSphereRandomVectorGenerator(dimension)),
             exponent,
             sharedSphere,
             noInterpolationTolerance);
    }

    
    public MicrosphereProjectionInterpolator(InterpolatingMicrosphere microsphere,
                                             double exponent,
                                             boolean sharedSphere,
                                             double noInterpolationTolerance)
        throws NotPositiveException {
        if (exponent < 0) {
            throw new NotPositiveException(exponent);
        }

        this.microsphere = microsphere;
        this.exponent = exponent;
        this.sharedSphere = sharedSphere;
        this.noInterpolationTolerance = noInterpolationTolerance;
    }

    
    public MultivariateFunction interpolate(final double[][] xval,
                                            final double[] yval)
        throws DimensionMismatchException,
               NoDataException,
               NullArgumentException {
        if (xval == null ||
            yval == null) {
            throw new NullArgumentException();
        }
        if (xval.length == 0) {
            throw new NoDataException();
        }
        if (xval.length != yval.length) {
            throw new DimensionMismatchException(xval.length, yval.length);
        }
        if (xval[0] == null) {
            throw new NullArgumentException();
        }
        final int dimension = microsphere.getDimension();
        if (dimension != xval[0].length) {
            throw new DimensionMismatchException(xval[0].length, dimension);
        }

        // Microsphere copy.
        final InterpolatingMicrosphere m = sharedSphere ? microsphere : microsphere.copy();

        return new MultivariateFunction() {
            
            public double value(double[] point) {
                return m.value(point,
                               xval,
                               yval,
                               exponent,
                               noInterpolationTolerance);
            }
        };
    }
}
