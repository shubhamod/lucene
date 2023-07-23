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

package org.apache.lucene.util.hnsw.math.optimization.direct;

import java.util.Comparator;

import org.apache.lucene.util.hnsw.math.optimization.PointValuePair;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;


@Deprecated
public class NelderMeadSimplex extends AbstractSimplex {
    
    private static final double DEFAULT_RHO = 1;
    
    private static final double DEFAULT_KHI = 2;
    
    private static final double DEFAULT_GAMMA = 0.5;
    
    private static final double DEFAULT_SIGMA = 0.5;
    
    private final double rho;
    
    private final double khi;
    
    private final double gamma;
    
    private final double sigma;

    
    public NelderMeadSimplex(final int n) {
        this(n, 1d);
    }

    
    public NelderMeadSimplex(final int n, double sideLength) {
        this(n, sideLength,
             DEFAULT_RHO, DEFAULT_KHI, DEFAULT_GAMMA, DEFAULT_SIGMA);
    }

    
    public NelderMeadSimplex(final int n, double sideLength,
                             final double rho, final double khi,
                             final double gamma, final double sigma) {
        super(n, sideLength);

        this.rho = rho;
        this.khi = khi;
        this.gamma = gamma;
        this.sigma = sigma;
    }

    
    public NelderMeadSimplex(final int n,
                             final double rho, final double khi,
                             final double gamma, final double sigma) {
        this(n, 1d, rho, khi, gamma, sigma);
    }

    
    public NelderMeadSimplex(final double[] steps) {
        this(steps, DEFAULT_RHO, DEFAULT_KHI, DEFAULT_GAMMA, DEFAULT_SIGMA);
    }

    
    public NelderMeadSimplex(final double[] steps,
                             final double rho, final double khi,
                             final double gamma, final double sigma) {
        super(steps);

        this.rho = rho;
        this.khi = khi;
        this.gamma = gamma;
        this.sigma = sigma;
    }

    
    public NelderMeadSimplex(final double[][] referenceSimplex) {
        this(referenceSimplex, DEFAULT_RHO, DEFAULT_KHI, DEFAULT_GAMMA, DEFAULT_SIGMA);
    }

    
    public NelderMeadSimplex(final double[][] referenceSimplex,
                             final double rho, final double khi,
                             final double gamma, final double sigma) {
        super(referenceSimplex);

        this.rho = rho;
        this.khi = khi;
        this.gamma = gamma;
        this.sigma = sigma;
    }

    
    @Override
    public void iterate(final MultivariateFunction evaluationFunction,
                        final Comparator<PointValuePair> comparator) {
        // The simplex has n + 1 points if dimension is n.
        final int n = getDimension();

        // Interesting values.
        final PointValuePair best = getPoint(0);
        final PointValuePair secondBest = getPoint(n - 1);
        final PointValuePair worst = getPoint(n);
        final double[] xWorst = worst.getPointRef();

        // Compute the centroid of the best vertices (dismissing the worst
        // point at index n).
        final double[] centroid = new double[n];
        for (int i = 0; i < n; i++) {
            final double[] x = getPoint(i).getPointRef();
            for (int j = 0; j < n; j++) {
                centroid[j] += x[j];
            }
        }
        final double scaling = 1.0 / n;
        for (int j = 0; j < n; j++) {
            centroid[j] *= scaling;
        }

        // compute the reflection point
        final double[] xR = new double[n];
        for (int j = 0; j < n; j++) {
            xR[j] = centroid[j] + rho * (centroid[j] - xWorst[j]);
        }
        final PointValuePair reflected
            = new PointValuePair(xR, evaluationFunction.value(xR), false);

        if (comparator.compare(best, reflected) <= 0 &&
            comparator.compare(reflected, secondBest) < 0) {
            // Accept the reflected point.
            replaceWorstPoint(reflected, comparator);
        } else if (comparator.compare(reflected, best) < 0) {
            // Compute the expansion point.
            final double[] xE = new double[n];
            for (int j = 0; j < n; j++) {
                xE[j] = centroid[j] + khi * (xR[j] - centroid[j]);
            }
            final PointValuePair expanded
                = new PointValuePair(xE, evaluationFunction.value(xE), false);

            if (comparator.compare(expanded, reflected) < 0) {
                // Accept the expansion point.
                replaceWorstPoint(expanded, comparator);
            } else {
                // Accept the reflected point.
                replaceWorstPoint(reflected, comparator);
            }
        } else {
            if (comparator.compare(reflected, worst) < 0) {
                // Perform an outside contraction.
                final double[] xC = new double[n];
                for (int j = 0; j < n; j++) {
                    xC[j] = centroid[j] + gamma * (xR[j] - centroid[j]);
                }
                final PointValuePair outContracted
                    = new PointValuePair(xC, evaluationFunction.value(xC), false);
                if (comparator.compare(outContracted, reflected) <= 0) {
                    // Accept the contraction point.
                    replaceWorstPoint(outContracted, comparator);
                    return;
                }
            } else {
                // Perform an inside contraction.
                final double[] xC = new double[n];
                for (int j = 0; j < n; j++) {
                    xC[j] = centroid[j] - gamma * (centroid[j] - xWorst[j]);
                }
                final PointValuePair inContracted
                    = new PointValuePair(xC, evaluationFunction.value(xC), false);

                if (comparator.compare(inContracted, worst) < 0) {
                    // Accept the contraction point.
                    replaceWorstPoint(inContracted, comparator);
                    return;
                }
            }

            // Perform a shrink.
            final double[] xSmallest = getPoint(0).getPointRef();
            for (int i = 1; i <= n; i++) {
                final double[] x = getPoint(i).getPoint();
                for (int j = 0; j < n; j++) {
                    x[j] = xSmallest[j] + sigma * (x[j] - xSmallest[j]);
                }
                setPoint(i, new PointValuePair(x, Double.NaN, false));
            }
            evaluate(evaluationFunction, comparator);
        }
    }
}
