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

import java.util.Arrays;
import java.util.Comparator;

import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.optimization.PointValuePair;
import org.apache.lucene.util.hnsw.math.optimization.OptimizationData;


@Deprecated
public abstract class AbstractSimplex implements OptimizationData {
    
    private PointValuePair[] simplex;
    
    private double[][] startConfiguration;
    
    private final int dimension;

    
    protected AbstractSimplex(int n) {
        this(n, 1d);
    }

    
    protected AbstractSimplex(int n,
                              double sideLength) {
        this(createHypercubeSteps(n, sideLength));
    }

    
    protected AbstractSimplex(final double[] steps) {
        if (steps == null) {
            throw new NullArgumentException();
        }
        if (steps.length == 0) {
            throw new ZeroException();
        }
        dimension = steps.length;

        // Only the relative position of the n final vertices with respect
        // to the first one are stored.
        startConfiguration = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            final double[] vertexI = startConfiguration[i];
            for (int j = 0; j < i + 1; j++) {
                if (steps[j] == 0) {
                    throw new ZeroException(LocalizedFormats.EQUAL_VERTICES_IN_SIMPLEX);
                }
                System.arraycopy(steps, 0, vertexI, 0, j + 1);
            }
        }
    }

    
    protected AbstractSimplex(final double[][] referenceSimplex) {
        if (referenceSimplex.length <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SIMPLEX_NEED_ONE_POINT,
                                                   referenceSimplex.length);
        }
        dimension = referenceSimplex.length - 1;

        // Only the relative position of the n final vertices with respect
        // to the first one are stored.
        startConfiguration = new double[dimension][dimension];
        final double[] ref0 = referenceSimplex[0];

        // Loop over vertices.
        for (int i = 0; i < referenceSimplex.length; i++) {
            final double[] refI = referenceSimplex[i];

            // Safety checks.
            if (refI.length != dimension) {
                throw new DimensionMismatchException(refI.length, dimension);
            }
            for (int j = 0; j < i; j++) {
                final double[] refJ = referenceSimplex[j];
                boolean allEquals = true;
                for (int k = 0; k < dimension; k++) {
                    if (refI[k] != refJ[k]) {
                        allEquals = false;
                        break;
                    }
                }
                if (allEquals) {
                    throw new MathIllegalArgumentException(LocalizedFormats.EQUAL_VERTICES_IN_SIMPLEX,
                                                           i, j);
                }
            }

            // Store vertex i position relative to vertex 0 position.
            if (i > 0) {
                final double[] confI = startConfiguration[i - 1];
                for (int k = 0; k < dimension; k++) {
                    confI[k] = refI[k] - ref0[k];
                }
            }
        }
    }

    
    public int getDimension() {
        return dimension;
    }

    
    public int getSize() {
        return simplex.length;
    }

    
    public abstract void iterate(final MultivariateFunction evaluationFunction,
                                 final Comparator<PointValuePair> comparator);

    
    public void build(final double[] startPoint) {
        if (dimension != startPoint.length) {
            throw new DimensionMismatchException(dimension, startPoint.length);
        }

        // Set first vertex.
        simplex = new PointValuePair[dimension + 1];
        simplex[0] = new PointValuePair(startPoint, Double.NaN);

        // Set remaining vertices.
        for (int i = 0; i < dimension; i++) {
            final double[] confI = startConfiguration[i];
            final double[] vertexI = new double[dimension];
            for (int k = 0; k < dimension; k++) {
                vertexI[k] = startPoint[k] + confI[k];
            }
            simplex[i + 1] = new PointValuePair(vertexI, Double.NaN);
        }
    }

    
    public void evaluate(final MultivariateFunction evaluationFunction,
                         final Comparator<PointValuePair> comparator) {
        // Evaluate the objective function at all non-evaluated simplex points.
        for (int i = 0; i < simplex.length; i++) {
            final PointValuePair vertex = simplex[i];
            final double[] point = vertex.getPointRef();
            if (Double.isNaN(vertex.getValue())) {
                simplex[i] = new PointValuePair(point, evaluationFunction.value(point), false);
            }
        }

        // Sort the simplex from best to worst.
        Arrays.sort(simplex, comparator);
    }

    
    protected void replaceWorstPoint(PointValuePair pointValuePair,
                                     final Comparator<PointValuePair> comparator) {
        for (int i = 0; i < dimension; i++) {
            if (comparator.compare(simplex[i], pointValuePair) > 0) {
                PointValuePair tmp = simplex[i];
                simplex[i] = pointValuePair;
                pointValuePair = tmp;
            }
        }
        simplex[dimension] = pointValuePair;
    }

    
    public PointValuePair[] getPoints() {
        final PointValuePair[] copy = new PointValuePair[simplex.length];
        System.arraycopy(simplex, 0, copy, 0, simplex.length);
        return copy;
    }

    
    public PointValuePair getPoint(int index) {
        if (index < 0 ||
            index >= simplex.length) {
            throw new OutOfRangeException(index, 0, simplex.length - 1);
        }
        return simplex[index];
    }

    
    protected void setPoint(int index, PointValuePair point) {
        if (index < 0 ||
            index >= simplex.length) {
            throw new OutOfRangeException(index, 0, simplex.length - 1);
        }
        simplex[index] = point;
    }

    
    protected void setPoints(PointValuePair[] points) {
        if (points.length != simplex.length) {
            throw new DimensionMismatchException(points.length, simplex.length);
        }
        simplex = points;
    }

    
    private static double[] createHypercubeSteps(int n,
                                                 double sideLength) {
        final double[] steps = new double[n];
        for (int i = 0; i < n; i++) {
            steps[i] = sideLength;
        }
        return steps;
    }
}
