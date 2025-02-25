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


public class RandomKeyMutation implements MutationPolicy {

    
    public Chromosome mutate(final Chromosome original) throws MathIllegalArgumentException {
        if (!(original instanceof RandomKey<?>)) {
            throw new MathIllegalArgumentException(LocalizedFormats.RANDOMKEY_MUTATION_WRONG_CLASS,
                                                   original.getClass().getSimpleName());
        }

        RandomKey<?> originalRk = (RandomKey<?>) original;
        List<Double> repr = originalRk.getRepresentation();
        int rInd = GeneticAlgorithm.getRandomGenerator().nextInt(repr.size());

        List<Double> newRepr = new ArrayList<Double> (repr);
        newRepr.set(rInd, GeneticAlgorithm.getRandomGenerator().nextDouble());

        return originalRk.newFixedLengthChromosome(newRepr);
    }

}
