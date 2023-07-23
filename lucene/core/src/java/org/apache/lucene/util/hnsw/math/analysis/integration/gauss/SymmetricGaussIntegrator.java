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
import org.apache.lucene.util.hnsw.math.util.Pair;


public class SymmetricGaussIntegrator extends GaussIntegrator {
    
    public SymmetricGaussIntegrator(double[] points,
                                    double[] weights)
        throws NonMonotonicSequenceException, DimensionMismatchException {
        super(points, weights);
    }

    
    public SymmetricGaussIntegrator(Pair<double[], double[]> pointsAndWeights)
        throws NonMonotonicSequenceException {
        this(pointsAndWeights.getFirst(), pointsAndWeights.getSecond());
    }

    
    @Override
    public double integrate(UnivariateFunction f) {
        final int ruleLength = getNumberOfPoints();

        if (ruleLength == 1) {
            return getWeight(0) * f.value(0d);
        }

        final int iMax = ruleLength / 2;
        double s = 0;
        double c = 0;
        for (int i = 0; i < iMax; i++) {
            final double p = getPoint(i);
            final double w = getWeight(i);

            final double f1 = f.value(p);
            final double f2 = f.value(-p);

            final double y = w * (f1 + f2) - c;
            final double t = s + y;

            c = (t - s) - y;
            s = t;
        }

        if (ruleLength % 2 != 0) {
            final double w = getWeight(iMax);

            final double y = w * f.value(0d) - c;
            final double t = s + y;

            s = t;
        }

        return s;
    }
}
