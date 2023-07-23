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

import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class HypergeometricDistribution extends AbstractIntegerDistribution {
    
    private static final long serialVersionUID = -436928820673516179L;
    
    private final int numberOfSuccesses;
    
    private final int populationSize;
    
    private final int sampleSize;
    
    private double numericalVariance = Double.NaN;
    
    private boolean numericalVarianceIsCalculated = false;

    
    public HypergeometricDistribution(int populationSize, int numberOfSuccesses, int sampleSize)
    throws NotPositiveException, NotStrictlyPositiveException, NumberIsTooLargeException {
        this(new Well19937c(), populationSize, numberOfSuccesses, sampleSize);
    }

    
    public HypergeometricDistribution(RandomGenerator rng,
                                      int populationSize,
                                      int numberOfSuccesses,
                                      int sampleSize)
    throws NotPositiveException, NotStrictlyPositiveException, NumberIsTooLargeException {
        super(rng);

        if (populationSize <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.POPULATION_SIZE,
                                                   populationSize);
        }
        if (numberOfSuccesses < 0) {
            throw new NotPositiveException(LocalizedFormats.NUMBER_OF_SUCCESSES,
                                           numberOfSuccesses);
        }
        if (sampleSize < 0) {
            throw new NotPositiveException(LocalizedFormats.NUMBER_OF_SAMPLES,
                                           sampleSize);
        }

        if (numberOfSuccesses > populationSize) {
            throw new NumberIsTooLargeException(LocalizedFormats.NUMBER_OF_SUCCESS_LARGER_THAN_POPULATION_SIZE,
                                                numberOfSuccesses, populationSize, true);
        }
        if (sampleSize > populationSize) {
            throw new NumberIsTooLargeException(LocalizedFormats.SAMPLE_SIZE_LARGER_THAN_POPULATION_SIZE,
                                                sampleSize, populationSize, true);
        }

        this.numberOfSuccesses = numberOfSuccesses;
        this.populationSize = populationSize;
        this.sampleSize = sampleSize;
    }

    
    public double cumulativeProbability(int x) {
        double ret;

        int[] domain = getDomain(populationSize, numberOfSuccesses, sampleSize);
        if (x < domain[0]) {
            ret = 0.0;
        } else if (x >= domain[1]) {
            ret = 1.0;
        } else {
            ret = innerCumulativeProbability(domain[0], x, 1);
        }

        return ret;
    }

    
    private int[] getDomain(int n, int m, int k) {
        return new int[] { getLowerDomain(n, m, k), getUpperDomain(m, k) };
    }

    
    private int getLowerDomain(int n, int m, int k) {
        return FastMath.max(0, m - (n - k));
    }

    
    public int getNumberOfSuccesses() {
        return numberOfSuccesses;
    }

    
    public int getPopulationSize() {
        return populationSize;
    }

    
    public int getSampleSize() {
        return sampleSize;
    }

    
    private int getUpperDomain(int m, int k) {
        return FastMath.min(k, m);
    }

    
    public double probability(int x) {
        final double logProbability = logProbability(x);
        return logProbability == Double.NEGATIVE_INFINITY ? 0 : FastMath.exp(logProbability);
    }

    
    @Override
    public double logProbability(int x) {
        double ret;

        int[] domain = getDomain(populationSize, numberOfSuccesses, sampleSize);
        if (x < domain[0] || x > domain[1]) {
            ret = Double.NEGATIVE_INFINITY;
        } else {
            double p = (double) sampleSize / (double) populationSize;
            double q = (double) (populationSize - sampleSize) / (double) populationSize;
            double p1 = SaddlePointExpansion.logBinomialProbability(x,
                    numberOfSuccesses, p, q);
            double p2 =
                    SaddlePointExpansion.logBinomialProbability(sampleSize - x,
                            populationSize - numberOfSuccesses, p, q);
            double p3 =
                    SaddlePointExpansion.logBinomialProbability(sampleSize, populationSize, p, q);
            ret = p1 + p2 - p3;
        }

        return ret;
    }

    
    public double upperCumulativeProbability(int x) {
        double ret;

        final int[] domain = getDomain(populationSize, numberOfSuccesses, sampleSize);
        if (x <= domain[0]) {
            ret = 1.0;
        } else if (x > domain[1]) {
            ret = 0.0;
        } else {
            ret = innerCumulativeProbability(domain[1], x, -1);
        }

        return ret;
    }

    
    private double innerCumulativeProbability(int x0, int x1, int dx) {
        double ret = probability(x0);
        while (x0 != x1) {
            x0 += dx;
            ret += probability(x0);
        }
        return ret;
    }

    
    public double getNumericalMean() {
        return getSampleSize() * (getNumberOfSuccesses() / (double) getPopulationSize());
    }

    
    public double getNumericalVariance() {
        if (!numericalVarianceIsCalculated) {
            numericalVariance = calculateNumericalVariance();
            numericalVarianceIsCalculated = true;
        }
        return numericalVariance;
    }

    
    protected double calculateNumericalVariance() {
        final double N = getPopulationSize();
        final double m = getNumberOfSuccesses();
        final double n = getSampleSize();
        return (n * m * (N - n) * (N - m)) / (N * N * (N - 1));
    }

    
    public int getSupportLowerBound() {
        return FastMath.max(0,
                            getSampleSize() + getNumberOfSuccesses() - getPopulationSize());
    }

    
    public int getSupportUpperBound() {
        return FastMath.min(getNumberOfSuccesses(), getSampleSize());
    }

    
    public boolean isSupportConnected() {
        return true;
    }
}
