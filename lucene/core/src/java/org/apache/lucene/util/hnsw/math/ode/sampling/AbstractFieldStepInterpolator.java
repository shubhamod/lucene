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

package org.apache.lucene.util.hnsw.math.ode.sampling;

import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.ode.FieldEquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;



public abstract class AbstractFieldStepInterpolator<T extends RealFieldElement<T>>
    implements FieldStepInterpolator<T> {

    
    private final FieldODEStateAndDerivative<T> globalPreviousState;

    
    private final FieldODEStateAndDerivative<T> globalCurrentState;

    
    private final FieldODEStateAndDerivative<T> softPreviousState;

    
    private final FieldODEStateAndDerivative<T> softCurrentState;

    
    private final boolean forward;

    
    private FieldEquationsMapper<T> mapper;

    
    protected AbstractFieldStepInterpolator(final boolean isForward,
                                            final FieldODEStateAndDerivative<T> globalPreviousState,
                                            final FieldODEStateAndDerivative<T> globalCurrentState,
                                            final FieldODEStateAndDerivative<T> softPreviousState,
                                            final FieldODEStateAndDerivative<T> softCurrentState,
                                            final FieldEquationsMapper<T> equationsMapper) {
        this.forward             = isForward;
        this.globalPreviousState = globalPreviousState;
        this.globalCurrentState  = globalCurrentState;
        this.softPreviousState   = softPreviousState;
        this.softCurrentState    = softCurrentState;
        this.mapper              = equationsMapper;
    }

    
    public AbstractFieldStepInterpolator<T> restrictStep(final FieldODEStateAndDerivative<T> previousState,
                                                         final FieldODEStateAndDerivative<T> currentState) {
        return create(forward, globalPreviousState, globalCurrentState, previousState, currentState, mapper);
    }

    
    protected abstract AbstractFieldStepInterpolator<T> create(boolean newForward,
                                                               FieldODEStateAndDerivative<T> newGlobalPreviousState,
                                                               FieldODEStateAndDerivative<T> newGlobalCurrentState,
                                                               FieldODEStateAndDerivative<T> newSoftPreviousState,
                                                               FieldODEStateAndDerivative<T> newSoftCurrentState,
                                                               FieldEquationsMapper<T> newMapper);

    
    public FieldODEStateAndDerivative<T> getGlobalPreviousState() {
        return globalPreviousState;
    }

    
    public FieldODEStateAndDerivative<T> getGlobalCurrentState() {
        return globalCurrentState;
    }

    
    public FieldODEStateAndDerivative<T> getPreviousState() {
        return softPreviousState;
    }

    
    public FieldODEStateAndDerivative<T> getCurrentState() {
        return softCurrentState;
    }

    
    public FieldODEStateAndDerivative<T> getInterpolatedState(final T time) {
        final T thetaH         = time.subtract(globalPreviousState.getTime());
        final T oneMinusThetaH = globalCurrentState.getTime().subtract(time);
        final T theta          = thetaH.divide(globalCurrentState.getTime().subtract(globalPreviousState.getTime()));
        return computeInterpolatedStateAndDerivatives(mapper, time, theta, thetaH, oneMinusThetaH);
    }

    
    public boolean isForward() {
        return forward;
    }

    
    protected abstract FieldODEStateAndDerivative<T> computeInterpolatedStateAndDerivatives(FieldEquationsMapper<T> equationsMapper,
                                                                                            T time, T theta,
                                                                                            T thetaH, T oneMinusThetaH)
        throws MaxCountExceededException;

}
