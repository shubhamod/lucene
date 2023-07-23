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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class CycleCrossover<T> implements CrossoverPolicy {

    
    private final boolean randomStart;

    
    public CycleCrossover() {
        this(false);
    }

    
    public CycleCrossover(final boolean randomStart) {
        this.randomStart = randomStart;
    }

    
    public boolean isRandomStart() {
        return randomStart;
    }

    
    @SuppressWarnings("unchecked")
    public ChromosomePair crossover(final Chromosome first, final Chromosome second)
        throws DimensionMismatchException, MathIllegalArgumentException {

        if (!(first instanceof AbstractListChromosome<?> && second instanceof AbstractListChromosome<?>)) {
            throw new MathIllegalArgumentException(LocalizedFormats.INVALID_FIXED_LENGTH_CHROMOSOME);
        }
        return mate((AbstractListChromosome<T>) first, (AbstractListChromosome<T>) second);
    }

    
    protected ChromosomePair mate(final AbstractListChromosome<T> first, final AbstractListChromosome<T> second)
        throws DimensionMismatchException {

        final int length = first.getLength();
        if (length != second.getLength()) {
            throw new DimensionMismatchException(second.getLength(), length);
        }

        // array representations of the parents
        final List<T> parent1Rep = first.getRepresentation();
        final List<T> parent2Rep = second.getRepresentation();
        // and of the children: do a crossover copy to simplify the later processing
        final List<T> child1Rep = new ArrayList<T>(second.getRepresentation());
        final List<T> child2Rep = new ArrayList<T>(first.getRepresentation());

        // the set of all visited indices so far
        final Set<Integer> visitedIndices = new HashSet<Integer>(length);
        // the indices of the current cycle
        final List<Integer> indices = new ArrayList<Integer>(length);

        // determine the starting index
        int idx = randomStart ? GeneticAlgorithm.getRandomGenerator().nextInt(length) : 0;
        int cycle = 1;

        while (visitedIndices.size() < length) {
            indices.add(idx);

            T item = parent2Rep.get(idx);
            idx = parent1Rep.indexOf(item);

            while (idx != indices.get(0)) {
                // add that index to the cycle indices
                indices.add(idx);
                // get the item in the second parent at that index
                item = parent2Rep.get(idx);
                // get the index of that item in the first parent
                idx = parent1Rep.indexOf(item);
            }

            // for even cycles: swap the child elements on the indices found in this cycle
            if (cycle++ % 2 != 0) {
                for (int i : indices) {
                    T tmp = child1Rep.get(i);
                    child1Rep.set(i, child2Rep.get(i));
                    child2Rep.set(i, tmp);
                }
            }

            visitedIndices.addAll(indices);
            // find next starting index: last one + 1 until we find an unvisited index
            idx = (indices.get(0) + 1) % length;
            while (visitedIndices.contains(idx) && visitedIndices.size() < length) {
                idx++;
                if (idx >= length) {
                    idx = 0;
                }
            }
            indices.clear();
        }

        return new ChromosomePair(first.newFixedLengthChromosome(child1Rep),
                                  second.newFixedLengthChromosome(child2Rep));
    }
}
