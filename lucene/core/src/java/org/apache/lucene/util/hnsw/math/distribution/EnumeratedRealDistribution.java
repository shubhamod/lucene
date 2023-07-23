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
package org.apache.lucene.util.hnsw.math.distribution;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NotANumberException;
import org.apache.lucene.util.hnsw.math.exception.NotFiniteNumberException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.Pair;


public class EnumeratedRealDistribution extends AbstractRealDistribution {

    
    private static final long serialVersionUID = 20130308L;

    
    protected final EnumeratedDistribution<Double> innerDistribution;

    
    public EnumeratedRealDistribution(final double[] singletons, final double[] probabilities)
    throws DimensionMismatchException, NotPositiveException, MathArithmeticException,
           NotFiniteNumberException, NotANumberException {
        this(new Well19937c(), singletons, probabilities);
    }

    
    public EnumeratedRealDistribution(final RandomGenerator rng,
                                    final double[] singletons, final double[] probabilities)
        throws DimensionMismatchException, NotPositiveException, MathArithmeticException,
               NotFiniteNumberException, NotANumberException {
        super(rng);

        innerDistribution = new EnumeratedDistribution<Double>(
                rng, createDistribution(singletons, probabilities));
    }

    
    public EnumeratedRealDistribution(final RandomGenerator rng, final double[] data) {
        super(rng);
        final Map<Double, Integer> dataMap = new HashMap<Double, Integer>();
        for (double value : data) {
            Integer count = dataMap.get(value);
            if (count == null) {
                count = 0;
            }
            dataMap.put(value, ++count);
        }
        final int massPoints = dataMap.size();
        final double denom = data.length;
        final double[] values = new double[massPoints];
        final double[] probabilities = new double[massPoints];
        int index = 0;
        for (Entry<Double, Integer> entry : dataMap.entrySet()) {
            values[index] = entry.getKey();
            probabilities[index] = entry.getValue().intValue() / denom;
            index++;
        }
        innerDistribution = new EnumeratedDistribution<Double>(rng, createDistribution(values, probabilities));
    }

    
    public EnumeratedRealDistribution(final double[] data) {
        this(new Well19937c(), data);
    }
    
    private static List<Pair<Double, Double>>  createDistribution(double[] singletons, double[] probabilities) {
        if (singletons.length != probabilities.length) {
            throw new DimensionMismatchException(probabilities.length, singletons.length);
        }

        final List<Pair<Double, Double>> samples = new ArrayList<Pair<Double, Double>>(singletons.length);

        for (int i = 0; i < singletons.length; i++) {
            samples.add(new Pair<Double, Double>(singletons[i], probabilities[i]));
        }
        return samples;

    }

    
    @Override
    public double probability(final double x) {
        return innerDistribution.probability(x);
    }

    
    public double density(final double x) {
        return probability(x);
    }

    
    public double cumulativeProbability(final double x) {
        double probability = 0;

        for (final Pair<Double, Double> sample : innerDistribution.getPmf()) {
            if (sample.getKey() <= x) {
                probability += sample.getValue();
            }
        }

        return probability;
    }

    
    @Override
    public double inverseCumulativeProbability(final double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }

        double probability = 0;
        double x = getSupportLowerBound();
        for (final Pair<Double, Double> sample : innerDistribution.getPmf()) {
            if (sample.getValue() == 0.0) {
                continue;
            }

            probability += sample.getValue();
            x = sample.getKey();

            if (probability >= p) {
                break;
            }
        }

        return x;
    }

    
    public double getNumericalMean() {
        double mean = 0;

        for (final Pair<Double, Double> sample : innerDistribution.getPmf()) {
            mean += sample.getValue() * sample.getKey();
        }

        return mean;
    }

    
    public double getNumericalVariance() {
        double mean = 0;
        double meanOfSquares = 0;

        for (final Pair<Double, Double> sample : innerDistribution.getPmf()) {
            mean += sample.getValue() * sample.getKey();
            meanOfSquares += sample.getValue() * sample.getKey() * sample.getKey();
        }

        return meanOfSquares - mean * mean;
    }

    
    public double getSupportLowerBound() {
        double min = Double.POSITIVE_INFINITY;
        for (final Pair<Double, Double> sample : innerDistribution.getPmf()) {
            if (sample.getKey() < min && sample.getValue() > 0) {
                min = sample.getKey();
            }
        }

        return min;
    }

    
    public double getSupportUpperBound() {
        double max = Double.NEGATIVE_INFINITY;
        for (final Pair<Double, Double> sample : innerDistribution.getPmf()) {
            if (sample.getKey() > max && sample.getValue() > 0) {
                max = sample.getKey();
            }
        }

        return max;
    }

    
    public boolean isSupportLowerBoundInclusive() {
        return true;
    }

    
    public boolean isSupportUpperBoundInclusive() {
        return true;
    }

    
    public boolean isSupportConnected() {
        return true;
    }

    
    @Override
    public double sample() {
        return innerDistribution.sample();
    }
}
