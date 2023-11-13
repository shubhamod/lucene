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

import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.util.MathArrays;




public class FieldExpandableODE<T extends RealFieldElement<T>> {

    
    private final FirstOrderFieldDifferentialEquations<T> primary;

    
    private List<FieldSecondaryEquations<T>> components;

    
    private FieldEquationsMapper<T> mapper;

    
    public FieldExpandableODE(final FirstOrderFieldDifferentialEquations<T> primary) {
        this.primary    = primary;
        this.components = new ArrayList<FieldSecondaryEquations<T>>();
        this.mapper     = new FieldEquationsMapper<T>(null, primary.getDimension());
    }

    
    public FieldEquationsMapper<T> getMapper() {
        return mapper;
    }

    
    public int addSecondaryEquations(final FieldSecondaryEquations<T> secondary) {

        components.add(secondary);
        mapper = new FieldEquationsMapper<T>(mapper, secondary.getDimension());

        return components.size();

    }

    
    public void init(final T t0, final T[] y0, final T finalTime) {

        // initialize primary equations
        int index = 0;
        final T[] primary0 = mapper.extractEquationData(index, y0);
        primary.init(t0, primary0, finalTime);

        // initialize secondary equations
        while (++index < mapper.getNumberOfEquations()) {
            final T[] secondary0 = mapper.extractEquationData(index, y0);
            components.get(index - 1).init(t0, primary0, secondary0, finalTime);
        }

    }

    
    public T[] computeDerivatives(final T t, final T[] y)
        throws MaxCountExceededException, DimensionMismatchException {

        final T[] yDot = MathArrays.buildArray(t.getField(), mapper.getTotalDimension());

        // compute derivatives of the primary equations
        int index = 0;
        final T[] primaryState    = mapper.extractEquationData(index, y);
        final T[] primaryStateDot = primary.computeDerivatives(t, primaryState);
        mapper.insertEquationData(index, primaryStateDot, yDot);

        // Add contribution for secondary equations
        while (++index < mapper.getNumberOfEquations()) {
            final T[] componentState    = mapper.extractEquationData(index, y);
            final T[] componentStateDot = components.get(index - 1).computeDerivatives(t, primaryState, primaryStateDot,
                                                                                       componentState);
            mapper.insertEquationData(index, componentStateDot, yDot);
        }

        return yDot;

    }

}
