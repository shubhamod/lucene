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
package org.apache.lucene.util.hnsw.math.util;

import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public abstract class ContinuedFraction {
    
    private static final double DEFAULT_EPSILON = 10e-9;

    
    protected ContinuedFraction() {
        super();
    }

    
    protected abstract double getA(int n, double x);

    
    protected abstract double getB(int n, double x);

    
    public double evaluate(double x) throws ConvergenceException {
        return evaluate(x, DEFAULT_EPSILON, Integer.MAX_VALUE);
    }

    
    public double evaluate(double x, double epsilon) throws ConvergenceException {
        return evaluate(x, epsilon, Integer.MAX_VALUE);
    }

    
    public double evaluate(double x, int maxIterations)
        throws ConvergenceException, MaxCountExceededException {
        return evaluate(x, DEFAULT_EPSILON, maxIterations);
    }

    
    public double evaluate(double x, double epsilon, int maxIterations)
        throws ConvergenceException, MaxCountExceededException {
        final double small = 1e-50;
        double hPrev = getA(0, x);

        // use the value of small as epsilon criteria for zero checks
        if (Precision.equals(hPrev, 0.0, small)) {
            hPrev = small;
        }

        int n = 1;
        double dPrev = 0.0;
        double cPrev = hPrev;
        double hN = hPrev;

        while (n < maxIterations) {
            final double a = getA(n, x);
            final double b = getB(n, x);

            double dN = a + b * dPrev;
            if (Precision.equals(dN, 0.0, small)) {
                dN = small;
            }
            double cN = a + b / cPrev;
            if (Precision.equals(cN, 0.0, small)) {
                cN = small;
            }

            dN = 1 / dN;
            final double deltaN = cN * dN;
            hN = hPrev * deltaN;

            if (Double.isInfinite(hN)) {
                throw new ConvergenceException(LocalizedFormats.CONTINUED_FRACTION_INFINITY_DIVERGENCE,
                                               x);
            }
            if (Double.isNaN(hN)) {
                throw new ConvergenceException(LocalizedFormats.CONTINUED_FRACTION_NAN_DIVERGENCE,
                                               x);
            }

            if (FastMath.abs(deltaN - 1.0) < epsilon) {
                break;
            }

            dPrev = dN;
            cPrev = cN;
            hPrev = hN;
            n++;
        }

        if (n >= maxIterations) {
            throw new MaxCountExceededException(LocalizedFormats.NON_CONVERGENT_CONTINUED_FRACTION,
                                                maxIterations, x);
        }

        return hN;
    }

}
