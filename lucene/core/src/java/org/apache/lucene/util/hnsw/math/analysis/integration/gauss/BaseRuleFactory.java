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

import java.util.Map;
import java.util.TreeMap;
import org.apache.lucene.util.hnsw.math.util.Pair;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public abstract class BaseRuleFactory<T extends Number> {
    
    private final Map<Integer, Pair<T[], T[]>> pointsAndWeights
        = new TreeMap<Integer, Pair<T[], T[]>>();
    
    private final Map<Integer, Pair<double[], double[]>> pointsAndWeightsDouble
        = new TreeMap<Integer, Pair<double[], double[]>>();

    
    public Pair<double[], double[]> getRule(int numberOfPoints)
        throws NotStrictlyPositiveException, DimensionMismatchException {

        if (numberOfPoints <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_POINTS,
                                                   numberOfPoints);
        }

        // Try to obtain the rule from the cache.
        Pair<double[], double[]> cached = pointsAndWeightsDouble.get(numberOfPoints);

        if (cached == null) {
            // Rule not computed yet.

            // Compute the rule.
            final Pair<T[], T[]> rule = getRuleInternal(numberOfPoints);
            cached = convertToDouble(rule);

            // Cache it.
            pointsAndWeightsDouble.put(numberOfPoints, cached);
        }

        // Return a copy.
        return new Pair<double[], double[]>(cached.getFirst().clone(),
                                            cached.getSecond().clone());
    }

    
    protected synchronized Pair<T[], T[]> getRuleInternal(int numberOfPoints)
        throws DimensionMismatchException {
        final Pair<T[], T[]> rule = pointsAndWeights.get(numberOfPoints);
        if (rule == null) {
            addRule(computeRule(numberOfPoints));
            // The rule should be available now.
            return getRuleInternal(numberOfPoints);
        }
        return rule;
    }

    
    protected void addRule(Pair<T[], T[]> rule) throws DimensionMismatchException {
        if (rule.getFirst().length != rule.getSecond().length) {
            throw new DimensionMismatchException(rule.getFirst().length,
                                                 rule.getSecond().length);
        }

        pointsAndWeights.put(rule.getFirst().length, rule);
    }

    
    protected abstract Pair<T[], T[]> computeRule(int numberOfPoints)
        throws DimensionMismatchException;

    
    private static <T extends Number> Pair<double[], double[]> convertToDouble(Pair<T[], T[]> rule) {
        final T[] pT = rule.getFirst();
        final T[] wT = rule.getSecond();

        final int len = pT.length;
        final double[] pD = new double[len];
        final double[] wD = new double[len];

        for (int i = 0; i < len; i++) {
            pD[i] = pT[i].doubleValue();
            wD[i] = wT[i].doubleValue();
        }

        return new Pair<double[], double[]>(pD, wD);
    }
}
