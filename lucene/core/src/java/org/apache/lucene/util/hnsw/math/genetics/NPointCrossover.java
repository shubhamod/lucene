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
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;


public class NPointCrossover<T> implements CrossoverPolicy {

    
    private final int crossoverPoints;

    
    public NPointCrossover(final int crossoverPoints) throws NotStrictlyPositiveException {
        if (crossoverPoints <= 0) {
            throw new NotStrictlyPositiveException(crossoverPoints);
        }
        this.crossoverPoints = crossoverPoints;
    }

    
    public int getCrossoverPoints() {
        return crossoverPoints;
    }

    
    @SuppressWarnings("unchecked") // OK because of instanceof checks
    public ChromosomePair crossover(final Chromosome first, final Chromosome second)
        throws DimensionMismatchException, MathIllegalArgumentException {

        if (!(first instanceof AbstractListChromosome<?> && second instanceof AbstractListChromosome<?>)) {
            throw new MathIllegalArgumentException(LocalizedFormats.INVALID_FIXED_LENGTH_CHROMOSOME);
        }
        return mate((AbstractListChromosome<T>) first, (AbstractListChromosome<T>) second);
    }

    
    private ChromosomePair mate(final AbstractListChromosome<T> first,
                                final AbstractListChromosome<T> second)
        throws DimensionMismatchException, NumberIsTooLargeException {

        final int length = first.getLength();
        if (length != second.getLength()) {
            throw new DimensionMismatchException(second.getLength(), length);
        }
        if (crossoverPoints >= length) {
            throw new NumberIsTooLargeException(crossoverPoints, length, false);
        }

        // array representations of the parents
        final List<T> parent1Rep = first.getRepresentation();
        final List<T> parent2Rep = second.getRepresentation();
        // and of the children
        final List<T> child1Rep = new ArrayList<T>(length);
        final List<T> child2Rep = new ArrayList<T>(length);

        final RandomGenerator random = GeneticAlgorithm.getRandomGenerator();

        List<T> c1 = child1Rep;
        List<T> c2 = child2Rep;

        int remainingPoints = crossoverPoints;
        int lastIndex = 0;
        for (int i = 0; i < crossoverPoints; i++, remainingPoints--) {
            // select the next crossover point at random
            final int crossoverIndex = 1 + lastIndex + random.nextInt(length - lastIndex - remainingPoints);

            // copy the current segment
            for (int j = lastIndex; j < crossoverIndex; j++) {
                c1.add(parent1Rep.get(j));
                c2.add(parent2Rep.get(j));
            }

            // swap the children for the next segment
            List<T> tmp = c1;
            c1 = c2;
            c2 = tmp;

            lastIndex = crossoverIndex;
        }

        // copy the last segment
        for (int j = lastIndex; j < length; j++) {
            c1.add(parent1Rep.get(j));
            c2.add(parent2Rep.get(j));
        }

        return new ChromosomePair(first.newFixedLengthChromosome(child1Rep),
                                  second.newFixedLengthChromosome(child2Rep));
    }
}
