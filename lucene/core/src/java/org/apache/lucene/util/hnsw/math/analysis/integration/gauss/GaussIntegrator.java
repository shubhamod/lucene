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
package org.apache.lucene.util.hnsw.math.analysis.integration.gauss;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.Pair;


public class GaussIntegrator {
    
    private final double[] points;
    
    private final double[] weights;

    
    public GaussIntegrator(double[] points,
                           double[] weights)
        throws NonMonotonicSequenceException, DimensionMismatchException {
        if (points.length != weights.length) {
            throw new DimensionMismatchException(points.length,
                                                 weights.length);
        }

        MathArrays.checkOrder(points, MathArrays.OrderDirection.INCREASING, true, true);

        this.points = points.clone();
        this.weights = weights.clone();
    }

    
    public GaussIntegrator(Pair<double[], double[]> pointsAndWeights)
        throws NonMonotonicSequenceException {
        this(pointsAndWeights.getFirst(), pointsAndWeights.getSecond());
    }

    
    public double integrate(UnivariateFunction f) {
        double s = 0;
        double c = 0;
        for (int i = 0; i < points.length; i++) {
            final double x = points[i];
            final double w = weights[i];
            final double y = w * f.value(x) - c;
            final double t = s + y;
            c = (t - s) - y;
            s = t;
        }
        return s;
    }

    
    public int getNumberOfPoints() {
        return points.length;
    }

    
    public double getPoint(int index) {
        return points[index];
    }

    
    public double getWeight(int index) {
        return weights[index];
    }
}
