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

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class TournamentSelection implements SelectionPolicy {

    
    private int arity;

    
    public TournamentSelection(final int arity) {
        this.arity = arity;
    }

    
    public ChromosomePair select(final Population population) throws MathIllegalArgumentException {
        return new ChromosomePair(tournament((ListPopulation) population),
                                  tournament((ListPopulation) population));
    }

    
    private Chromosome tournament(final ListPopulation population) throws MathIllegalArgumentException {
        if (population.getPopulationSize() < this.arity) {
            throw new MathIllegalArgumentException(LocalizedFormats.TOO_LARGE_TOURNAMENT_ARITY,
                                                   arity, population.getPopulationSize());
        }
        // auxiliary population
        ListPopulation tournamentPopulation = new ListPopulation(this.arity) {
            
            public Population nextGeneration() {
                // not useful here
                return null;
            }
        };

        // create a copy of the chromosome list
        List<Chromosome> chromosomes = new ArrayList<Chromosome> (population.getChromosomes());
        for (int i=0; i<this.arity; i++) {
            // select a random individual and add it to the tournament
            int rind = GeneticAlgorithm.getRandomGenerator().nextInt(chromosomes.size());
            tournamentPopulation.addChromosome(chromosomes.get(rind));
            // do not select it again
            chromosomes.remove(rind);
        }
        // the winner takes it all
        return tournamentPopulation.getFittestChromosome();
    }

    
    public int getArity() {
        return arity;
    }

    
    public void setArity(final int arity) {
        this.arity = arity;
    }

}
