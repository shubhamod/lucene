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

import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.JDKRandomGenerator;


public class GeneticAlgorithm {

    
    //@GuardedBy("this")
    private static RandomGenerator randomGenerator = new JDKRandomGenerator();

    
    private final CrossoverPolicy crossoverPolicy;

    
    private final double crossoverRate;

    
    private final MutationPolicy mutationPolicy;

    
    private final double mutationRate;

    
    private final SelectionPolicy selectionPolicy;

    
    private int generationsEvolved = 0;

    
    public GeneticAlgorithm(final CrossoverPolicy crossoverPolicy,
                            final double crossoverRate,
                            final MutationPolicy mutationPolicy,
                            final double mutationRate,
                            final SelectionPolicy selectionPolicy) throws OutOfRangeException {

        if (crossoverRate < 0 || crossoverRate > 1) {
            throw new OutOfRangeException(LocalizedFormats.CROSSOVER_RATE,
                                          crossoverRate, 0, 1);
        }
        if (mutationRate < 0 || mutationRate > 1) {
            throw new OutOfRangeException(LocalizedFormats.MUTATION_RATE,
                                          mutationRate, 0, 1);
        }
        this.crossoverPolicy = crossoverPolicy;
        this.crossoverRate = crossoverRate;
        this.mutationPolicy = mutationPolicy;
        this.mutationRate = mutationRate;
        this.selectionPolicy = selectionPolicy;
    }

    
    public static synchronized void setRandomGenerator(final RandomGenerator random) {
        randomGenerator = random;
    }

    
    public static synchronized RandomGenerator getRandomGenerator() {
        return randomGenerator;
    }

    
    public Population evolve(final Population initial, final StoppingCondition condition) {
        Population current = initial;
        generationsEvolved = 0;
        while (!condition.isSatisfied(current)) {
            current = nextGeneration(current);
            generationsEvolved++;
        }
        return current;
    }

    
    public Population nextGeneration(final Population current) {
        Population nextGeneration = current.nextGeneration();

        RandomGenerator randGen = getRandomGenerator();

        while (nextGeneration.getPopulationSize() < nextGeneration.getPopulationLimit()) {
            // select parent chromosomes
            ChromosomePair pair = getSelectionPolicy().select(current);

            // crossover?
            if (randGen.nextDouble() < getCrossoverRate()) {
                // apply crossover policy to create two offspring
                pair = getCrossoverPolicy().crossover(pair.getFirst(), pair.getSecond());
            }

            // mutation?
            if (randGen.nextDouble() < getMutationRate()) {
                // apply mutation policy to the chromosomes
                pair = new ChromosomePair(
                    getMutationPolicy().mutate(pair.getFirst()),
                    getMutationPolicy().mutate(pair.getSecond()));
            }

            // add the first chromosome to the population
            nextGeneration.addChromosome(pair.getFirst());
            // is there still a place for the second chromosome?
            if (nextGeneration.getPopulationSize() < nextGeneration.getPopulationLimit()) {
                // add the second chromosome to the population
                nextGeneration.addChromosome(pair.getSecond());
            }
        }

        return nextGeneration;
    }

    
    public CrossoverPolicy getCrossoverPolicy() {
        return crossoverPolicy;
    }

    
    public double getCrossoverRate() {
        return crossoverRate;
    }

    
    public MutationPolicy getMutationPolicy() {
        return mutationPolicy;
    }

    
    public double getMutationRate() {
        return mutationRate;
    }

    
    public SelectionPolicy getSelectionPolicy() {
        return selectionPolicy;
    }

    
    public int getGenerationsEvolved() {
        return generationsEvolved;
    }

}
