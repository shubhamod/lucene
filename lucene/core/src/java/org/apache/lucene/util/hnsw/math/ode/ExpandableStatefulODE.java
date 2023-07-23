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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;




public class ExpandableStatefulODE {

    
    private final FirstOrderDifferentialEquations primary;

    
    private final EquationsMapper primaryMapper;

    
    private double time;

    
    private final double[] primaryState;

    
    private final double[] primaryStateDot;

    
    private List<SecondaryComponent> components;

    
    public ExpandableStatefulODE(final FirstOrderDifferentialEquations primary) {
        final int n          = primary.getDimension();
        this.primary         = primary;
        this.primaryMapper   = new EquationsMapper(0, n);
        this.time            = Double.NaN;
        this.primaryState    = new double[n];
        this.primaryStateDot = new double[n];
        this.components      = new ArrayList<SecondaryComponent>();
    }

    
    public FirstOrderDifferentialEquations getPrimary() {
        return primary;
    }

    
    public int getTotalDimension() {
        if (components.isEmpty()) {
            // there are no secondary equations, the complete set is limited to the primary set
            return primaryMapper.getDimension();
        } else {
            // there are secondary equations, the complete set ends after the last set
            final EquationsMapper lastMapper = components.get(components.size() - 1).mapper;
            return lastMapper.getFirstIndex() + lastMapper.getDimension();
        }
    }

    
    public void computeDerivatives(final double t, final double[] y, final double[] yDot)
        throws MaxCountExceededException, DimensionMismatchException {

        // compute derivatives of the primary equations
        primaryMapper.extractEquationData(y, primaryState);
        primary.computeDerivatives(t, primaryState, primaryStateDot);

        // Add contribution for secondary equations
        for (final SecondaryComponent component : components) {
            component.mapper.extractEquationData(y, component.state);
            component.equation.computeDerivatives(t, primaryState, primaryStateDot,
                                                  component.state, component.stateDot);
            component.mapper.insertEquationData(component.stateDot, yDot);
        }

        primaryMapper.insertEquationData(primaryStateDot, yDot);

    }

    
    public int addSecondaryEquations(final SecondaryEquations secondary) {

        final int firstIndex;
        if (components.isEmpty()) {
            // lazy creation of the components list
            components = new ArrayList<SecondaryComponent>();
            firstIndex = primary.getDimension();
        } else {
            final SecondaryComponent last = components.get(components.size() - 1);
            firstIndex = last.mapper.getFirstIndex() + last.mapper.getDimension();
        }

        components.add(new SecondaryComponent(secondary, firstIndex));

        return components.size() - 1;

    }

    
    public EquationsMapper getPrimaryMapper() {
        return primaryMapper;
    }

    
    public EquationsMapper[] getSecondaryMappers() {
        final EquationsMapper[] mappers = new EquationsMapper[components.size()];
        for (int i = 0; i < mappers.length; ++i) {
            mappers[i] = components.get(i).mapper;
        }
        return mappers;
    }

    
    public void setTime(final double time) {
        this.time = time;
    }

    
    public double getTime() {
        return time;
    }

    
    public void setPrimaryState(final double[] primaryState) throws DimensionMismatchException {

        // safety checks
        if (primaryState.length != this.primaryState.length) {
            throw new DimensionMismatchException(primaryState.length, this.primaryState.length);
        }

        // set the data
        System.arraycopy(primaryState, 0, this.primaryState, 0, primaryState.length);

    }

    
    public double[] getPrimaryState() {
        return primaryState.clone();
    }

    
    public double[] getPrimaryStateDot() {
        return primaryStateDot.clone();
    }

    
    public void setSecondaryState(final int index, final double[] secondaryState)
        throws DimensionMismatchException {

        // get either the secondary state
        double[] localArray = components.get(index).state;

        // safety checks
        if (secondaryState.length != localArray.length) {
            throw new DimensionMismatchException(secondaryState.length, localArray.length);
        }

        // set the data
        System.arraycopy(secondaryState, 0, localArray, 0, secondaryState.length);

    }

    
    public double[] getSecondaryState(final int index) {
        return components.get(index).state.clone();
    }

    
    public double[] getSecondaryStateDot(final int index) {
        return components.get(index).stateDot.clone();
    }

    
    public void setCompleteState(final double[] completeState)
        throws DimensionMismatchException {

        // safety checks
        if (completeState.length != getTotalDimension()) {
            throw new DimensionMismatchException(completeState.length, getTotalDimension());
        }

        // set the data
        primaryMapper.extractEquationData(completeState, primaryState);
        for (final SecondaryComponent component : components) {
            component.mapper.extractEquationData(completeState, component.state);
        }

    }

    
    public double[] getCompleteState() throws DimensionMismatchException {

        // allocate complete array
        double[] completeState = new double[getTotalDimension()];

        // set the data
        primaryMapper.insertEquationData(primaryState, completeState);
        for (final SecondaryComponent component : components) {
            component.mapper.insertEquationData(component.state, completeState);
        }

        return completeState;

    }

    
    private static class SecondaryComponent {

        
        private final SecondaryEquations equation;

        
        private final EquationsMapper mapper;

        
        private final double[] state;

        
        private final double[] stateDot;

        
        SecondaryComponent(final SecondaryEquations equation, final int firstIndex) {
            final int n   = equation.getDimension();
            this.equation = equation;
            mapper        = new EquationsMapper(firstIndex, n);
            state         = new double[n];
            stateDot      = new double[n];
        }

    }

}
