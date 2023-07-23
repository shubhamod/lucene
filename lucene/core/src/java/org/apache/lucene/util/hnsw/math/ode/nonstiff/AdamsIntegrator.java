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

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NoBracketingException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.ode.ExpandableStatefulODE;
import org.apache.lucene.util.hnsw.math.ode.MultistepIntegrator;



public abstract class AdamsIntegrator extends MultistepIntegrator {

    
    private final AdamsNordsieckTransformer transformer;

    
    public AdamsIntegrator(final String name, final int nSteps, final int order,
                           final double minStep, final double maxStep,
                           final double scalAbsoluteTolerance,
                           final double scalRelativeTolerance)
        throws NumberIsTooSmallException {
        super(name, nSteps, order, minStep, maxStep,
              scalAbsoluteTolerance, scalRelativeTolerance);
        transformer = AdamsNordsieckTransformer.getInstance(nSteps);
    }

    
    public AdamsIntegrator(final String name, final int nSteps, final int order,
                           final double minStep, final double maxStep,
                           final double[] vecAbsoluteTolerance,
                           final double[] vecRelativeTolerance)
        throws IllegalArgumentException {
        super(name, nSteps, order, minStep, maxStep,
              vecAbsoluteTolerance, vecRelativeTolerance);
        transformer = AdamsNordsieckTransformer.getInstance(nSteps);
    }

    
    @Override
    public abstract void integrate(final ExpandableStatefulODE equations, final double t)
        throws NumberIsTooSmallException, DimensionMismatchException,
               MaxCountExceededException, NoBracketingException;

    
    @Override
    protected Array2DRowRealMatrix initializeHighOrderDerivatives(final double h, final double[] t,
                                                                  final double[][] y,
                                                                  final double[][] yDot) {
        return transformer.initializeHighOrderDerivatives(h, t, y, yDot);
    }

    
    public Array2DRowRealMatrix updateHighOrderDerivativesPhase1(final Array2DRowRealMatrix highOrder) {
        return transformer.updateHighOrderDerivativesPhase1(highOrder);
    }

    
    public void updateHighOrderDerivativesPhase2(final double[] start,
                                                 final double[] end,
                                                 final Array2DRowRealMatrix highOrder) {
        transformer.updateHighOrderDerivativesPhase2(start, end, highOrder);
    }

}
