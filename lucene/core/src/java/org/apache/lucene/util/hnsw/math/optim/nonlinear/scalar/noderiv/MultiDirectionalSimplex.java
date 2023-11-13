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
package org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.noderiv;

import java.util.Comparator;

import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.optim.PointValuePair;


public class MultiDirectionalSimplex extends AbstractSimplex {
    
    private static final double DEFAULT_KHI = 2;
    
    private static final double DEFAULT_GAMMA = 0.5;
    
    private final double khi;
    
    private final double gamma;

    
    public MultiDirectionalSimplex(final int n) {
        this(n, 1d);
    }

    
    public MultiDirectionalSimplex(final int n, double sideLength) {
        this(n, sideLength, DEFAULT_KHI, DEFAULT_GAMMA);
    }

    
    public MultiDirectionalSimplex(final int n,
                                   final double khi, final double gamma) {
        this(n, 1d, khi, gamma);
    }

    
    public MultiDirectionalSimplex(final int n, double sideLength,
                                   final double khi, final double gamma) {
        super(n, sideLength);

        this.khi   = khi;
        this.gamma = gamma;
    }

    
    public MultiDirectionalSimplex(final double[] steps) {
        this(steps, DEFAULT_KHI, DEFAULT_GAMMA);
    }

    
    public MultiDirectionalSimplex(final double[] steps,
                                   final double khi, final double gamma) {
        super(steps);

        this.khi   = khi;
        this.gamma = gamma;
    }

    
    public MultiDirectionalSimplex(final double[][] referenceSimplex) {
        this(referenceSimplex, DEFAULT_KHI, DEFAULT_GAMMA);
    }

    
    public MultiDirectionalSimplex(final double[][] referenceSimplex,
                                   final double khi, final double gamma) {
        super(referenceSimplex);

        this.khi   = khi;
        this.gamma = gamma;
    }

    
    @Override
    public void iterate(final MultivariateFunction evaluationFunction,
                        final Comparator<PointValuePair> comparator) {
        // Save the original simplex.
        final PointValuePair[] original = getPoints();
        final PointValuePair best = original[0];

        // Perform a reflection step.
        final PointValuePair reflected = evaluateNewSimplex(evaluationFunction,
                                                                original, 1, comparator);
        if (comparator.compare(reflected, best) < 0) {
            // Compute the expanded simplex.
            final PointValuePair[] reflectedSimplex = getPoints();
            final PointValuePair expanded = evaluateNewSimplex(evaluationFunction,
                                                                   original, khi, comparator);
            if (comparator.compare(reflected, expanded) <= 0) {
                // Keep the reflected simplex.
                setPoints(reflectedSimplex);
            }
            // Keep the expanded simplex.
            return;
        }

        // Compute the contracted simplex.
        evaluateNewSimplex(evaluationFunction, original, gamma, comparator);

    }

    
    private PointValuePair evaluateNewSimplex(final MultivariateFunction evaluationFunction,
                                                  final PointValuePair[] original,
                                                  final double coeff,
                                                  final Comparator<PointValuePair> comparator) {
        final double[] xSmallest = original[0].getPointRef();
        // Perform a linear transformation on all the simplex points,
        // except the first one.
        setPoint(0, original[0]);
        final int dim = getDimension();
        for (int i = 1; i < getSize(); i++) {
            final double[] xOriginal = original[i].getPointRef();
            final double[] xTransformed = new double[dim];
            for (int j = 0; j < dim; j++) {
                xTransformed[j] = xSmallest[j] + coeff * (xSmallest[j] - xOriginal[j]);
            }
            setPoint(i, new PointValuePair(xTransformed, Double.NaN, false));
        }

        // Evaluate the simplex.
        evaluate(evaluationFunction, comparator);

        return getPoint(0);
    }
}
