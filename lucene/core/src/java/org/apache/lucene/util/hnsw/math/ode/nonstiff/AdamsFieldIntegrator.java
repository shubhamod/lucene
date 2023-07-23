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

package org.apache.lucene.util.hnsw.math.ode.nonstiff;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowFieldMatrix;
import org.apache.lucene.util.hnsw.math.ode.FieldExpandableODE;
import org.apache.lucene.util.hnsw.math.ode.FieldODEState;
import org.apache.lucene.util.hnsw.math.ode.FieldODEStateAndDerivative;
import org.apache.lucene.util.hnsw.math.ode.MultistepFieldIntegrator;



public abstract class AdamsFieldIntegrator<T extends RealFieldElement<T>> extends MultistepFieldIntegrator<T> {

    
    private final AdamsNordsieckFieldTransformer<T> transformer;

    
    public AdamsFieldIntegrator(final Field<T> field, final String name,
                                final int nSteps, final int order,
                                final double minStep, final double maxStep,
                                final double scalAbsoluteTolerance,
                                final double scalRelativeTolerance)
        throws NumberIsTooSmallException {
        super(field, name, nSteps, order, minStep, maxStep,
              scalAbsoluteTolerance, scalRelativeTolerance);
        transformer = AdamsNordsieckFieldTransformer.getInstance(field, nSteps);
    }

    
    public AdamsFieldIntegrator(final Field<T> field, final String name,
                                final int nSteps, final int order,
                                final double minStep, final double maxStep,
                                final double[] vecAbsoluteTolerance,
                                final double[] vecRelativeTolerance)
        throws IllegalArgumentException {
        super(field, name, nSteps, order, minStep, maxStep,
              vecAbsoluteTolerance, vecRelativeTolerance);
        transformer = AdamsNordsieckFieldTransformer.getInstance(field, nSteps);
    }

    
    public abstract FieldODEStateAndDerivative<T> integrate(final FieldExpandableODE<T> equations,
                                                            final FieldODEState<T> initialState,
                                                            final T finalTime)
        throws NumberIsTooSmallException, DimensionMismatchException,
               MaxCountExceededException, NoBracketingException;

    
    @Override
    protected Array2DRowFieldMatrix<T> initializeHighOrderDerivatives(final T h, final T[] t,
                                                                      final T[][] y,
                                                                      final T[][] yDot) {
        return transformer.initializeHighOrderDerivatives(h, t, y, yDot);
    }

    
    public Array2DRowFieldMatrix<T> updateHighOrderDerivativesPhase1(final Array2DRowFieldMatrix<T> highOrder) {
        return transformer.updateHighOrderDerivativesPhase1(highOrder);
    }

    
    public void updateHighOrderDerivativesPhase2(final T[] start, final T[] end,
                                                 final Array2DRowFieldMatrix<T> highOrder) {
        transformer.updateHighOrderDerivativesPhase2(start, end, highOrder);
    }

}
