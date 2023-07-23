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

import java.math.BigDecimal;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.util.Pair;


public class GaussIntegratorFactory {
    
    private final BaseRuleFactory<Double> legendre = new LegendreRuleFactory();
    
    private final BaseRuleFactory<BigDecimal> legendreHighPrecision = new LegendreHighPrecisionRuleFactory();
    
    private final BaseRuleFactory<Double> hermite = new HermiteRuleFactory();

    
    public GaussIntegrator legendre(int numberOfPoints) {
        return new GaussIntegrator(getRule(legendre, numberOfPoints));
    }

    
    public GaussIntegrator legendre(int numberOfPoints,
                                    double lowerBound,
                                    double upperBound)
        throws NotStrictlyPositiveException {
        return new GaussIntegrator(transform(getRule(legendre, numberOfPoints),
                                             lowerBound, upperBound));
    }

    
    public GaussIntegrator legendreHighPrecision(int numberOfPoints)
        throws NotStrictlyPositiveException {
        return new GaussIntegrator(getRule(legendreHighPrecision, numberOfPoints));
    }

    
    public GaussIntegrator legendreHighPrecision(int numberOfPoints,
                                                 double lowerBound,
                                                 double upperBound)
        throws NotStrictlyPositiveException {
        return new GaussIntegrator(transform(getRule(legendreHighPrecision, numberOfPoints),
                                             lowerBound, upperBound));
    }

    
    public SymmetricGaussIntegrator hermite(int numberOfPoints) {
        return new SymmetricGaussIntegrator(getRule(hermite, numberOfPoints));
    }

    
    private static Pair<double[], double[]> getRule(BaseRuleFactory<? extends Number> factory,
                                                    int numberOfPoints)
        throws NotStrictlyPositiveException, DimensionMismatchException {
        return factory.getRule(numberOfPoints);
    }

    
    private static Pair<double[], double[]> transform(Pair<double[], double[]> rule,
                                                      double a,
                                                      double b) {
        final double[] points = rule.getFirst();
        final double[] weights = rule.getSecond();

        // Scaling
        final double scale = (b - a) / 2;
        final double shift = a + scale;

        for (int i = 0; i < points.length; i++) {
            points[i] = points[i] * scale + shift;
            weights[i] *= scale;
        }

        return new Pair<double[], double[]>(points, weights);
    }
}
