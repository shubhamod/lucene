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

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NotANumberException;
import org.apache.lucene.util.hnsw.math.exception.NotFiniteNumberException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.Pair;


public class EnumeratedDistribution<T> implements Serializable {

    
    private static final long serialVersionUID = 20123308L;

    
    protected final RandomGenerator random;

    
    private final List<T> singletons;

    
    private final double[] probabilities;

    
    private final double[] cumulativeProbabilities;

    
    public EnumeratedDistribution(final List<Pair<T, Double>> pmf)
        throws NotPositiveException, MathArithmeticException, NotFiniteNumberException, NotANumberException {
        this(new Well19937c(), pmf);
    }

    
    public EnumeratedDistribution(final RandomGenerator rng, final List<Pair<T, Double>> pmf)
        throws NotPositiveException, MathArithmeticException, NotFiniteNumberException, NotANumberException {
        random = rng;

        singletons = new ArrayList<T>(pmf.size());
        final double[] probs = new double[pmf.size()];

        for (int i = 0; i < pmf.size(); i++) {
            final Pair<T, Double> sample = pmf.get(i);
            singletons.add(sample.getKey());
            final double p = sample.getValue();
            if (p < 0) {
                throw new NotPositiveException(sample.getValue());
            }
            if (Double.isInfinite(p)) {
                throw new NotFiniteNumberException(p);
            }
            if (Double.isNaN(p)) {
                throw new NotANumberException();
            }
            probs[i] = p;
        }

        probabilities = MathArrays.normalizeArray(probs, 1.0);

        cumulativeProbabilities = new double[probabilities.length];
        double sum = 0;
        for (int i = 0; i < probabilities.length; i++) {
            sum += probabilities[i];
            cumulativeProbabilities[i] = sum;
        }
    }

    
    public void reseedRandomGenerator(long seed) {
        random.setSeed(seed);
    }

    
    double probability(final T x) {
        double probability = 0;

        for (int i = 0; i < probabilities.length; i++) {
            if ((x == null && singletons.get(i) == null) ||
                (x != null && x.equals(singletons.get(i)))) {
                probability += probabilities[i];
            }
        }

        return probability;
    }

    
    public List<Pair<T, Double>> getPmf() {
        final List<Pair<T, Double>> samples = new ArrayList<Pair<T, Double>>(probabilities.length);

        for (int i = 0; i < probabilities.length; i++) {
            samples.add(new Pair<T, Double>(singletons.get(i), probabilities[i]));
        }

        return samples;
    }

    
    public T sample() {
        final double randomValue = random.nextDouble();

        int index = Arrays.binarySearch(cumulativeProbabilities, randomValue);
        if (index < 0) {
            index = -index-1;
        }

        if (index >= 0 &&
            index < probabilities.length &&
            randomValue < cumulativeProbabilities[index]) {
            return singletons.get(index);
        }

        /* This should never happen, but it ensures we will return a correct
         * object in case there is some floating point inequality problem
         * wrt the cumulative probabilities. */
        return singletons.get(singletons.size() - 1);
    }

    
    public Object[] sample(int sampleSize) throws NotStrictlyPositiveException {
        if (sampleSize <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_SAMPLES,
                    sampleSize);
        }

        final Object[] out = new Object[sampleSize];

        for (int i = 0; i < sampleSize; i++) {
            out[i] = sample();
        }

        return out;

    }

    
    public T[] sample(int sampleSize, final T[] array) throws NotStrictlyPositiveException {
        if (sampleSize <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_SAMPLES, sampleSize);
        }

        if (array == null) {
            throw new NullArgumentException(LocalizedFormats.INPUT_ARRAY);
        }

        T[] out;
        if (array.length < sampleSize) {
            @SuppressWarnings("unchecked") // safe as both are of type T
            final T[] unchecked = (T[]) Array.newInstance(array.getClass().getComponentType(), sampleSize);
            out = unchecked;
        } else {
            out = array;
        }

        for (int i = 0; i < sampleSize; i++) {
            out[i] = sample();
        }

        return out;

    }

}
