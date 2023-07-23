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


public abstract class Chromosome implements Comparable<Chromosome>,Fitness {
    
    private static final double NO_FITNESS = Double.NEGATIVE_INFINITY;

    
    private double fitness = NO_FITNESS;

    
    public double getFitness() {
        if (this.fitness == NO_FITNESS) {
            // no cache - compute the fitness
            this.fitness = fitness();
        }
        return this.fitness;
    }

    
    public int compareTo(final Chromosome another) {
        return Double.compare(getFitness(), another.getFitness());
    }

    
    protected boolean isSame(final Chromosome another) {
        return false;
    }

    
    protected Chromosome findSameChromosome(final Population population) {
        for (Chromosome anotherChr : population) {
            if (this.isSame(anotherChr)) {
                return anotherChr;
            }
        }
        return null;
    }

    
    public void searchForFitnessUpdate(final Population population) {
        Chromosome sameChromosome = findSameChromosome(population);
        if (sameChromosome != null) {
            fitness = sameChromosome.getFitness();
        }
    }

}
