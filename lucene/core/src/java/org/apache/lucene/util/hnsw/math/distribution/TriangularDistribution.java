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

package org.apache.lucene.util.hnsw.math.distribution;

import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.random.Well19937c;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class TriangularDistribution extends AbstractRealDistribution {
    
    private static final long serialVersionUID = 20120112L;
    
    private final double a;
    
    private final double b;
    
    private final double c;
    
    private final double solverAbsoluteAccuracy;

    
    public TriangularDistribution(double a, double c, double b)
        throws NumberIsTooLargeException, NumberIsTooSmallException {
        this(new Well19937c(), a, c, b);
    }

    
    public TriangularDistribution(RandomGenerator rng,
                                  double a,
                                  double c,
                                  double b)
        throws NumberIsTooLargeException, NumberIsTooSmallException {
        super(rng);

        if (a >= b) {
            throw new NumberIsTooLargeException(
                            LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND,
                            a, b, false);
        }
        if (c < a) {
            throw new NumberIsTooSmallException(
                    LocalizedFormats.NUMBER_TOO_SMALL, c, a, true);
        }
        if (c > b) {
            throw new NumberIsTooLargeException(
                    LocalizedFormats.NUMBER_TOO_LARGE, c, b, true);
        }

        this.a = a;
        this.c = c;
        this.b = b;
        solverAbsoluteAccuracy = FastMath.max(FastMath.ulp(a), FastMath.ulp(b));
    }

    
    public double getMode() {
        return c;
    }

    
    @Override
    protected double getSolverAbsoluteAccuracy() {
        return solverAbsoluteAccuracy;
    }

    
    public double density(double x) {
        if (x < a) {
            return 0;
        }
        if (a <= x && x < c) {
            double divident = 2 * (x - a);
            double divisor = (b - a) * (c - a);
            return divident / divisor;
        }
        if (x == c) {
            return 2 / (b - a);
        }
        if (c < x && x <= b) {
            double divident = 2 * (b - x);
            double divisor = (b - a) * (b - c);
            return divident / divisor;
        }
        return 0;
    }

    
    public double cumulativeProbability(double x)  {
        if (x < a) {
            return 0;
        }
        if (a <= x && x < c) {
            double divident = (x - a) * (x - a);
            double divisor = (b - a) * (c - a);
            return divident / divisor;
        }
        if (x == c) {
            return (c - a) / (b - a);
        }
        if (c < x && x <= b) {
            double divident = (b - x) * (b - x);
            double divisor = (b - a) * (b - c);
            return 1 - (divident / divisor);
        }
        return 1;
    }

    
    public double getNumericalMean() {
        return (a + b + c) / 3;
    }

    
    public double getNumericalVariance() {
        return (a * a + b * b + c * c - a * b - a * c - b * c) / 18;
    }

    
    public double getSupportLowerBound() {
        return a;
    }

    
    public double getSupportUpperBound() {
        return b;
    }

    
    public boolean isSupportLowerBoundInclusive() {
        return true;
    }

    
    public boolean isSupportUpperBoundInclusive() {
        return true;
    }

    
    public boolean isSupportConnected() {
        return true;
    }

    
    @Override
    public double inverseCumulativeProbability(double p)
        throws OutOfRangeException {
        if (p < 0 || p > 1) {
            throw new OutOfRangeException(p, 0, 1);
        }
        if (p == 0) {
            return a;
        }
        if (p == 1) {
            return b;
        }
        if (p < (c - a) / (b - a)) {
            return a + FastMath.sqrt(p * (b - a) * (c - a));
        }
        return b - FastMath.sqrt((1 - p) * (b - a) * (b - c));
    }
}
