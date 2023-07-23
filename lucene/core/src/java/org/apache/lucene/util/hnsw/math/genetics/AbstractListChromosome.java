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
import java.util.List;


public abstract class AbstractListChromosome<T> extends Chromosome {

    
    private final List<T> representation;

    
    public AbstractListChromosome(final List<T> representation) throws InvalidRepresentationException {
        this(representation, true);
    }

    
    public AbstractListChromosome(final T[] representation) throws InvalidRepresentationException {
        this(Arrays.asList(representation));
    }

    
    public AbstractListChromosome(final List<T> representation, final boolean copyList) {
        checkValidity(representation);
        this.representation =
                Collections.unmodifiableList(copyList ? new ArrayList<T>(representation) : representation);
    }

    
    protected abstract void checkValidity(List<T> chromosomeRepresentation) throws InvalidRepresentationException;

    
    protected List<T> getRepresentation() {
        return representation;
    }

    
    public int getLength() {
        return getRepresentation().size();
    }

    
    public abstract AbstractListChromosome<T> newFixedLengthChromosome(final List<T> chromosomeRepresentation);

    
    @Override
    public String toString() {
        return String.format("(f=%s %s)", getFitness(), getRepresentation());
    }
}
