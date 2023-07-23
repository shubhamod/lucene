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

import java.util.Collection;

import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.analysis.solvers.BracketedRealFieldUnivariateSolver;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.ode.events.FieldEventHandler;
import org.apache.lucene.util.hnsw.math.ode.sampling.FieldStepHandler;



public interface FirstOrderFieldIntegrator<T extends RealFieldElement<T>> {

    
    String getName();

    
    void addStepHandler(FieldStepHandler<T> handler);

    
    Collection<FieldStepHandler<T>> getStepHandlers();

    
    void clearStepHandlers();

    
    void addEventHandler(FieldEventHandler<T>  handler, double maxCheckInterval,
                         double convergence, int maxIterationCount);

    
    void addEventHandler(FieldEventHandler<T>  handler, double maxCheckInterval,
                         double convergence, int maxIterationCount,
                         BracketedRealFieldUnivariateSolver<T> solver);

    
    Collection<FieldEventHandler<T> > getEventHandlers();

    
    void clearEventHandlers();

    
    FieldODEStateAndDerivative<T> getCurrentStepStart();

    
    T getCurrentSignedStepsize();

    
    void setMaxEvaluations(int maxEvaluations);

    
    int getMaxEvaluations();

    
    int getEvaluations();

    
    FieldODEStateAndDerivative<T> integrate(FieldExpandableODE<T> equations,
                                            FieldODEState<T> initialState, T finalTime)
        throws NumberIsTooSmallException, MaxCountExceededException, NoBracketingException;

}
