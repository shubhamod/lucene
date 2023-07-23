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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.util.MathArrays;



public class FieldODEState<T extends RealFieldElement<T>> {

    
    private final T time;

    
    private final T[] state;

    
    private final T[][] secondaryState;

    
    public FieldODEState(T time, T[] state) {
        this(time, state, null);
    }

    
    public FieldODEState(T time, T[] state, T[][] secondaryState) {
        this.time           = time;
        this.state          = state.clone();
        this.secondaryState = copy(time.getField(), secondaryState);
    }

    
    protected T[][] copy(final Field<T> field, final T[][] original) {

        // special handling of null arrays
        if (original == null) {
            return null;
        }

        // allocate the array
        final T[][] copied = MathArrays.buildArray(field, original.length, -1);

        // copy content
        for (int i = 0; i < original.length; ++i) {
            copied[i] = original[i].clone();
        }

        return copied;

    }

    
    public T getTime() {
        return time;
    }

    
    public int getStateDimension() {
        return state.length;
    }

    
    public T[] getState() {
        return state.clone();
    }

    
    public int getNumberOfSecondaryStates() {
        return secondaryState == null ? 0 : secondaryState.length;
    }

    
    public int getSecondaryStateDimension(final int index) {
        return index == 0 ? state.length : secondaryState[index - 1].length;
    }

    
    public T[] getSecondaryState(final int index) {
        return index == 0 ? state.clone() : secondaryState[index - 1].clone();
    }

}
