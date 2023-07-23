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
package org.apache.lucene.util.hnsw.math.genetics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public abstract class RandomKey<T> extends AbstractListChromosome<Double> implements PermutationChromosome<T> {

    
    private final List<Double> sortedRepresentation;

    
    private final List<Integer> baseSeqPermutation;

    
    public RandomKey(final List<Double> representation) throws InvalidRepresentationException {
        super(representation);
        // store the sorted representation
        List<Double> sortedRepr = new ArrayList<Double> (getRepresentation());
        Collections.sort(sortedRepr);
        sortedRepresentation = Collections.unmodifiableList(sortedRepr);
        // store the permutation of [0,1,...,n-1] list for toString() and isSame() methods
        baseSeqPermutation = Collections.unmodifiableList(
            decodeGeneric(baseSequence(getLength()), getRepresentation(), sortedRepresentation)
        );
    }

    
    public RandomKey(final Double[] representation) throws InvalidRepresentationException {
        this(Arrays.asList(representation));
    }

    
    public List<T> decode(final List<T> sequence) {
        return decodeGeneric(sequence, getRepresentation(), sortedRepresentation);
    }

    
    private static <S> List<S> decodeGeneric(final List<S> sequence, List<Double> representation,
                                             final List<Double> sortedRepr)
        throws DimensionMismatchException {

        int l = sequence.size();

        // the size of the three lists must be equal
        if (representation.size() != l) {
            throw new DimensionMismatchException(representation.size(), l);
        }
        if (sortedRepr.size() != l) {
            throw new DimensionMismatchException(sortedRepr.size(), l);
        }

        // do not modify the original representation
        List<Double> reprCopy = new ArrayList<Double> (representation);

        // now find the indices in the original repr and use them for permuting
        List<S> res = new ArrayList<S> (l);
        for (int i=0; i<l; i++) {
            int index = reprCopy.indexOf(sortedRepr.get(i));
            res.add(sequence.get(index));
            reprCopy.set(index, null);
        }
        return res;
    }

    
    @Override
    protected boolean isSame(final Chromosome another) {
        // type check
        if (! (another instanceof RandomKey<?>)) {
            return false;
        }
        RandomKey<?> anotherRk = (RandomKey<?>) another;
        // size check
        if (getLength() != anotherRk.getLength()) {
            return false;
        }

        // two different representations can still encode the same permutation
        // the ordering is what counts
        List<Integer> thisPerm = this.baseSeqPermutation;
        List<Integer> anotherPerm = anotherRk.baseSeqPermutation;

        for (int i=0; i<getLength(); i++) {
            if (thisPerm.get(i) != anotherPerm.get(i)) {
                return false;
            }
        }
        // the permutations are the same
        return true;
    }

    
    @Override
    protected void checkValidity(final List<Double> chromosomeRepresentation)
        throws InvalidRepresentationException {

        for (double val : chromosomeRepresentation) {
            if (val < 0 || val > 1) {
                throw new InvalidRepresentationException(LocalizedFormats.OUT_OF_RANGE_SIMPLE,
                                                         val, 0, 1);
            }
        }
    }


    
    public static final List<Double> randomPermutation(final int l) {
        List<Double> repr = new ArrayList<Double>(l);
        for (int i=0; i<l; i++) {
            repr.add(GeneticAlgorithm.getRandomGenerator().nextDouble());
        }
        return repr;
    }

    
    public static final List<Double> identityPermutation(final int l) {
        List<Double> repr = new ArrayList<Double>(l);
        for (int i=0; i<l; i++) {
            repr.add((double)i/l);
        }
        return repr;
    }

    
    public static <S> List<Double> comparatorPermutation(final List<S> data,
                                                         final Comparator<S> comparator) {
        List<S> sortedData = new ArrayList<S>(data);
        Collections.sort(sortedData, comparator);

        return inducedPermutation(data, sortedData);
    }

    
    public static <S> List<Double> inducedPermutation(final List<S> originalData,
                                                      final List<S> permutedData)
        throws DimensionMismatchException, MathIllegalArgumentException {

        if (originalData.size() != permutedData.size()) {
            throw new DimensionMismatchException(permutedData.size(), originalData.size());
        }
        int l = originalData.size();

        List<S> origDataCopy = new ArrayList<S> (originalData);

        Double[] res = new Double[l];
        for (int i=0; i<l; i++) {
            int index = origDataCopy.indexOf(permutedData.get(i));
            if (index == -1) {
                throw new MathIllegalArgumentException(LocalizedFormats.DIFFERENT_ORIG_AND_PERMUTED_DATA);
            }
            res[index] = (double) i / l;
            origDataCopy.set(index, null);
        }
        return Arrays.asList(res);
    }

    
    @Override
    public String toString() {
        return String.format("(f=%s pi=(%s))", getFitness(), baseSeqPermutation);
    }

    
    private static List<Integer> baseSequence(final int l) {
        List<Integer> baseSequence = new ArrayList<Integer> (l);
        for (int i=0; i<l; i++) {
            baseSequence.add(i);
        }
        return baseSequence;
    }
}
