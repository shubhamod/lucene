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
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;


public abstract class ListPopulation implements Population {

    
    private List<Chromosome> chromosomes;

    
    private int populationLimit;

    
    public ListPopulation(final int populationLimit) throws NotPositiveException {
        this(Collections.<Chromosome> emptyList(), populationLimit);
    }

    
    public ListPopulation(final List<Chromosome> chromosomes, final int populationLimit)
        throws NullArgumentException, NotPositiveException, NumberIsTooLargeException {

        if (chromosomes == null) {
            throw new NullArgumentException();
        }
        if (populationLimit <= 0) {
            throw new NotPositiveException(LocalizedFormats.POPULATION_LIMIT_NOT_POSITIVE, populationLimit);
        }
        if (chromosomes.size() > populationLimit) {
            throw new NumberIsTooLargeException(LocalizedFormats.LIST_OF_CHROMOSOMES_BIGGER_THAN_POPULATION_SIZE,
                                                chromosomes.size(), populationLimit, false);
        }
        this.populationLimit = populationLimit;
        this.chromosomes = new ArrayList<Chromosome>(populationLimit);
        this.chromosomes.addAll(chromosomes);
    }

    
    @Deprecated
    public void setChromosomes(final List<Chromosome> chromosomes)
        throws NullArgumentException, NumberIsTooLargeException {

        if (chromosomes == null) {
            throw new NullArgumentException();
        }
        if (chromosomes.size() > populationLimit) {
            throw new NumberIsTooLargeException(LocalizedFormats.LIST_OF_CHROMOSOMES_BIGGER_THAN_POPULATION_SIZE,
                                                chromosomes.size(), populationLimit, false);
        }
        this.chromosomes.clear();
        this.chromosomes.addAll(chromosomes);
    }

    
    public void addChromosomes(final Collection<Chromosome> chromosomeColl) throws NumberIsTooLargeException {
        if (chromosomes.size() + chromosomeColl.size() > populationLimit) {
            throw new NumberIsTooLargeException(LocalizedFormats.LIST_OF_CHROMOSOMES_BIGGER_THAN_POPULATION_SIZE,
                                                chromosomes.size(), populationLimit, false);
        }
        this.chromosomes.addAll(chromosomeColl);
    }

    
    public List<Chromosome> getChromosomes() {
        return Collections.unmodifiableList(chromosomes);
    }

    
    protected List<Chromosome> getChromosomeList() {
        return chromosomes;
    }

    
    public void addChromosome(final Chromosome chromosome) throws NumberIsTooLargeException {
        if (chromosomes.size() >= populationLimit) {
            throw new NumberIsTooLargeException(LocalizedFormats.LIST_OF_CHROMOSOMES_BIGGER_THAN_POPULATION_SIZE,
                                                chromosomes.size(), populationLimit, false);
        }
        this.chromosomes.add(chromosome);
    }

    
    public Chromosome getFittestChromosome() {
        // best so far
        Chromosome bestChromosome = this.chromosomes.get(0);
        for (Chromosome chromosome : this.chromosomes) {
            if (chromosome.compareTo(bestChromosome) > 0) {
                // better chromosome found
                bestChromosome = chromosome;
            }
        }
        return bestChromosome;
    }

    
    public int getPopulationLimit() {
        return this.populationLimit;
    }

    
    public void setPopulationLimit(final int populationLimit) throws NotPositiveException, NumberIsTooSmallException {
        if (populationLimit <= 0) {
            throw new NotPositiveException(LocalizedFormats.POPULATION_LIMIT_NOT_POSITIVE, populationLimit);
        }
        if (populationLimit < chromosomes.size()) {
            throw new NumberIsTooSmallException(populationLimit, chromosomes.size(), true);
        }
        this.populationLimit = populationLimit;
    }

    
    public int getPopulationSize() {
        return this.chromosomes.size();
    }

    
    @Override
    public String toString() {
        return this.chromosomes.toString();
    }

    
    public Iterator<Chromosome> iterator() {
        return getChromosomes().iterator();
    }
}
