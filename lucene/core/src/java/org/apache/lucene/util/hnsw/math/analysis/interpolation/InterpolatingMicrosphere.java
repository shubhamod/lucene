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

import java.util.List;
import java.util.ArrayList;
import org.apache.lucene.util.hnsw.math.random.UnitSphereRandomVectorGenerator;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class InterpolatingMicrosphere {
    
    private final List<Facet> microsphere;
    
    private final List<FacetData> microsphereData;
    
    private final int dimension;
    
    private final int size;
    
    private final double maxDarkFraction;
    
    private final double darkThreshold;
    
    private final double background;

    
    protected InterpolatingMicrosphere(int dimension,
                                       int size,
                                       double maxDarkFraction,
                                       double darkThreshold,
                                       double background) {
        if (dimension <= 0) {
            throw new NotStrictlyPositiveException(dimension);
        }
        if (size <= 0) {
            throw new NotStrictlyPositiveException(size);
        }
        if (maxDarkFraction < 0 ||
            maxDarkFraction > 1) {
            throw new OutOfRangeException(maxDarkFraction, 0, 1);
        }
        if (darkThreshold < 0) {
            throw new NotPositiveException(darkThreshold);
        }

        this.dimension = dimension;
        this.size = size;
        this.maxDarkFraction = maxDarkFraction;
        this.darkThreshold = darkThreshold;
        this.background = background;
        microsphere = new ArrayList<Facet>(size);
        microsphereData = new ArrayList<FacetData>(size);
    }

    
    public InterpolatingMicrosphere(int dimension,
                                    int size,
                                    double maxDarkFraction,
                                    double darkThreshold,
                                    double background,
                                    UnitSphereRandomVectorGenerator rand) {
        this(dimension, size, maxDarkFraction, darkThreshold, background);

        // Generate the microsphere normals, assuming that a number of
        // randomly generated normals will represent a sphere.
        for (int i = 0; i < size; i++) {
            add(rand.nextVector(), false);
        }
    }

    
    protected InterpolatingMicrosphere(InterpolatingMicrosphere other) {
        dimension = other.dimension;
        size = other.size;
        maxDarkFraction = other.maxDarkFraction;
        darkThreshold = other.darkThreshold;
        background = other.background;

        // Field can be shared.
        microsphere = other.microsphere;

        // Field must be copied.
        microsphereData = new ArrayList<FacetData>(size);
        for (FacetData fd : other.microsphereData) {
            microsphereData.add(new FacetData(fd.illumination(), fd.sample()));
        }
    }

    
    public InterpolatingMicrosphere copy() {
        return new InterpolatingMicrosphere(this);
    }

    
    public int getDimension() {
        return dimension;
    }

    
    public int getSize() {
        return size;
    }

    
    public double value(double[] point,
                        double[][] samplePoints,
                        double[] sampleValues,
                        double exponent,
                        double noInterpolationTolerance) {
        if (exponent < 0) {
            throw new NotPositiveException(exponent);
        }

        clear();

        // Contribution of each sample point to the illumination of the
        // microsphere's facets.
        final int numSamples = samplePoints.length;
        for (int i = 0; i < numSamples; i++) {
            // Vector between interpolation point and current sample point.
            final double[] diff = MathArrays.ebeSubtract(samplePoints[i], point);
            final double diffNorm = MathArrays.safeNorm(diff);

            if (FastMath.abs(diffNorm) < noInterpolationTolerance) {
                // No need to interpolate, as the interpolation point is
                // actually (very close to) one of the sampled points.
                return sampleValues[i];
            }

            final double weight = FastMath.pow(diffNorm, -exponent);
            illuminate(diff, sampleValues[i], weight);
        }

        return interpolate();
    }

    
    protected void add(double[] normal,
                       boolean copy) {
        if (microsphere.size() >= size) {
            throw new MaxCountExceededException(size);
        }
        if (normal.length > dimension) {
            throw new DimensionMismatchException(normal.length, dimension);
        }

        microsphere.add(new Facet(copy ? normal.clone() : normal));
        microsphereData.add(new FacetData(0d, 0d));
    }

    
    private double interpolate() {
        // Number of non-illuminated facets.
        int darkCount = 0;

        double value = 0;
        double totalWeight = 0;
        for (FacetData fd : microsphereData) {
            final double iV = fd.illumination();
            if (iV != 0d) {
                value += iV * fd.sample();
                totalWeight += iV;
            } else {
                ++darkCount;
            }
        }

        final double darkFraction = darkCount / (double) size;

        return darkFraction <= maxDarkFraction ?
            value / totalWeight :
            background;
    }

    
    private void illuminate(double[] sampleDirection,
                            double sampleValue,
                            double weight) {
        for (int i = 0; i < size; i++) {
            final double[] n = microsphere.get(i).getNormal();
            final double cos = MathArrays.cosAngle(n, sampleDirection);

            if (cos > 0) {
                final double illumination = cos * weight;

                if (illumination > darkThreshold &&
                    illumination > microsphereData.get(i).illumination()) {
                    microsphereData.set(i, new FacetData(illumination, sampleValue));
                }
            }
        }
    }

    
    private void clear() {
        for (int i = 0; i < size; i++) {
            microsphereData.set(i, new FacetData(0d, 0d));
        }
    }

    
    private static class Facet {
        
        private final double[] normal;

        
        Facet(double[] n) {
            normal = n;
        }

        
        public double[] getNormal() {
            return normal;
        }
    }

    
    private static class FacetData {
        
        private final double illumination;
        
        private final double sample;

        
        FacetData(double illumination, double sample) {
            this.illumination = illumination;
            this.sample = sample;
        }

        
        public double illumination() {
            return illumination;
        }

        
        public double sample() {
            return sample;
        }
    }
}
