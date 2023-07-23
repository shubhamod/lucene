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

import java.util.Collections;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class ElitisticListPopulation extends ListPopulation {

    
    private double elitismRate = 0.9;

    
    public ElitisticListPopulation(final List<Chromosome> chromosomes, final int populationLimit,
                                   final double elitismRate)
        throws NullArgumentException, NotPositiveException, NumberIsTooLargeException, OutOfRangeException {

        super(chromosomes, populationLimit);
        setElitismRate(elitismRate);
    }

    
    public ElitisticListPopulation(final int populationLimit, final double elitismRate)
        throws NotPositiveException, OutOfRangeException {

        super(populationLimit);
        setElitismRate(elitismRate);
    }

    
    public Population nextGeneration() {
        // initialize a new generation with the same parameters
        ElitisticListPopulation nextGeneration =
                new ElitisticListPopulation(getPopulationLimit(), getElitismRate());

        final List<Chromosome> oldChromosomes = getChromosomeList();
        Collections.sort(oldChromosomes);

        // index of the last "not good enough" chromosome
        int boundIndex = (int) FastMath.ceil((1.0 - getElitismRate()) * oldChromosomes.size());
        for (int i = boundIndex; i < oldChromosomes.size(); i++) {
            nextGeneration.addChromosome(oldChromosomes.get(i));
        }
        return nextGeneration;
    }

    
    public void setElitismRate(final double elitismRate) throws OutOfRangeException {
        if (elitismRate < 0 || elitismRate > 1) {
            throw new OutOfRangeException(LocalizedFormats.ELITISM_RATE, elitismRate, 0, 1);
        }
        this.elitismRate = elitismRate;
    }

    
    public double getElitismRate() {
        return this.elitismRate;
    }

}
