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

package org.apache.lucene.util.hnsw.math.ode;

import org.apache.lucene.util.hnsw.math.RealFieldElement;



public class FieldODEStateAndDerivative<T extends RealFieldElement<T>> extends FieldODEState<T> {

    
    private final T[] derivative;

    
    private final T[][] secondaryDerivative;

    
    public FieldODEStateAndDerivative(T time, T[] state, T[] derivative) {
        this(time, state, derivative, null, null);
    }

    
    public FieldODEStateAndDerivative(T time, T[] state, T[] derivative, T[][] secondaryState, T[][] secondaryDerivative) {
        super(time, state, secondaryState);
        this.derivative          = derivative.clone();
        this.secondaryDerivative = copy(time.getField(), secondaryDerivative);
    }

    
    public T[] getDerivative() {
        return derivative.clone();
    }

    
    public T[] getSecondaryDerivative(final int index) {
        return index == 0 ? derivative.clone() : secondaryDerivative[index - 1].clone();
    }

}
