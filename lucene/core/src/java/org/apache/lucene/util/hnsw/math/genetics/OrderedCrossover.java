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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class OrderedCrossover<T> implements CrossoverPolicy {

    
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
        // and of the children
        final List<T> child1 = new ArrayList<T>(length);
        final List<T> child2 = new ArrayList<T>(length);
        // sets of already inserted items for quick access
        final Set<T> child1Set = new HashSet<T>(length);
        final Set<T> child2Set = new HashSet<T>(length);

        final RandomGenerator random = GeneticAlgorithm.getRandomGenerator();
        // choose random points, making sure that lb < ub.
        int a = random.nextInt(length);
        int b;
        do {
            b = random.nextInt(length);
        } while (a == b);
        // determine the lower and upper bounds
        final int lb = FastMath.min(a, b);
        final int ub = FastMath.max(a, b);

        // add the subLists that are between lb and ub
        child1.addAll(parent1Rep.subList(lb, ub + 1));
        child1Set.addAll(child1);
        child2.addAll(parent2Rep.subList(lb, ub + 1));
        child2Set.addAll(child2);

        // iterate over every item in the parents
        for (int i = 1; i <= length; i++) {
            final int idx = (ub + i) % length;

            // retrieve the current item in each parent
            final T item1 = parent1Rep.get(idx);
            final T item2 = parent2Rep.get(idx);

            // if the first child already contains the item in the second parent add it
            if (!child1Set.contains(item2)) {
                child1.add(item2);
                child1Set.add(item2);
            }

            // if the second child already contains the item in the first parent add it
            if (!child2Set.contains(item1)) {
                child2.add(item1);
                child2Set.add(item1);
            }
        }

        // rotate so that the original slice is in the same place as in the parents.
        Collections.rotate(child1, lb);
        Collections.rotate(child2, lb);

        return new ChromosomePair(first.newFixedLengthChromosome(child1),
                                  second.newFixedLengthChromosome(child2));
    }
}
