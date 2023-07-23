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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.random.UnitSphereRandomVectorGenerator;
import org.apache.lucene.util.hnsw.math.util.FastMath;


@Deprecated
public class MicrosphereInterpolatingFunction
    implements MultivariateFunction {
    
    private final int dimension;
    
    private final List<MicrosphereSurfaceElement> microsphere;
    
    private final double brightnessExponent;
    
    private final Map<RealVector, Double> samples;

    
    private static class MicrosphereSurfaceElement {
        
        private final RealVector normal;
        
        private double brightestIllumination;
        
        private Map.Entry<RealVector, Double> brightestSample;

        
        MicrosphereSurfaceElement(double[] n) {
            normal = new ArrayRealVector(n);
        }

        
        RealVector normal() {
            return normal;
        }

        
        void reset() {
            brightestIllumination = 0;
            brightestSample = null;
        }

        
        void store(final double illuminationFromSample,
                   final Map.Entry<RealVector, Double> sample) {
            if (illuminationFromSample > this.brightestIllumination) {
                this.brightestIllumination = illuminationFromSample;
                this.brightestSample = sample;
            }
        }

        
        double illumination() {
            return brightestIllumination;
        }

        
        Map.Entry<RealVector, Double> sample() {
            return brightestSample;
        }
    }

    
    public MicrosphereInterpolatingFunction(double[][] xval,
                                            double[] yval,
                                            int brightnessExponent,
                                            int microsphereElements,
                                            UnitSphereRandomVectorGenerator rand)
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

        dimension = xval[0].length;
        this.brightnessExponent = brightnessExponent;

        // Copy data samples.
        samples = new HashMap<RealVector, Double>(yval.length);
        for (int i = 0; i < xval.length; ++i) {
            final double[] xvalI = xval[i];
            if (xvalI == null) {
                throw new NullArgumentException();
            }
            if (xvalI.length != dimension) {
                throw new DimensionMismatchException(xvalI.length, dimension);
            }

            samples.put(new ArrayRealVector(xvalI), yval[i]);
        }

        microsphere = new ArrayList<MicrosphereSurfaceElement>(microsphereElements);
        // Generate the microsphere, assuming that a fairly large number of
        // randomly generated normals will represent a sphere.
        for (int i = 0; i < microsphereElements; i++) {
            microsphere.add(new MicrosphereSurfaceElement(rand.nextVector()));
        }
    }

    
    public double value(double[] point) throws DimensionMismatchException {
        final RealVector p = new ArrayRealVector(point);

        // Reset.
        for (MicrosphereSurfaceElement md : microsphere) {
            md.reset();
        }

        // Compute contribution of each sample points to the microsphere elements illumination
        for (Map.Entry<RealVector, Double> sd : samples.entrySet()) {

            // Vector between interpolation point and current sample point.
            final RealVector diff = sd.getKey().subtract(p);
            final double diffNorm = diff.getNorm();

            if (FastMath.abs(diffNorm) < FastMath.ulp(1d)) {
                // No need to interpolate, as the interpolation point is
                // actually (very close to) one of the sampled points.
                return sd.getValue();
            }

            for (MicrosphereSurfaceElement md : microsphere) {
                final double w = FastMath.pow(diffNorm, -brightnessExponent);
                md.store(cosAngle(diff, md.normal()) * w, sd);
            }

        }

        // Interpolation calculation.
        double value = 0;
        double totalWeight = 0;
        for (MicrosphereSurfaceElement md : microsphere) {
            final double iV = md.illumination();
            final Map.Entry<RealVector, Double> sd = md.sample();
            if (sd != null) {
                value += iV * sd.getValue();
                totalWeight += iV;
            }
        }

        return value / totalWeight;
    }

    
    private double cosAngle(final RealVector v, final RealVector w) {
        return v.dotProduct(w) / (v.getNorm() * w.getNorm());
    }
}
