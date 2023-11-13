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
package org.apache.lucene.util.hnsw.math.analysis.integration;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;


@Deprecated
public class LegendreGaussIntegrator extends BaseAbstractUnivariateIntegrator {

    
    private static final double[] ABSCISSAS_2 = {
        -1.0 / FastMath.sqrt(3.0),
         1.0 / FastMath.sqrt(3.0)
    };

    
    private static final double[] WEIGHTS_2 = {
        1.0,
        1.0
    };

    
    private static final double[] ABSCISSAS_3 = {
        -FastMath.sqrt(0.6),
         0.0,
         FastMath.sqrt(0.6)
    };

    
    private static final double[] WEIGHTS_3 = {
        5.0 / 9.0,
        8.0 / 9.0,
        5.0 / 9.0
    };

    
    private static final double[] ABSCISSAS_4 = {
        -FastMath.sqrt((15.0 + 2.0 * FastMath.sqrt(30.0)) / 35.0),
        -FastMath.sqrt((15.0 - 2.0 * FastMath.sqrt(30.0)) / 35.0),
         FastMath.sqrt((15.0 - 2.0 * FastMath.sqrt(30.0)) / 35.0),
         FastMath.sqrt((15.0 + 2.0 * FastMath.sqrt(30.0)) / 35.0)
    };

    
    private static final double[] WEIGHTS_4 = {
        (90.0 - 5.0 * FastMath.sqrt(30.0)) / 180.0,
        (90.0 + 5.0 * FastMath.sqrt(30.0)) / 180.0,
        (90.0 + 5.0 * FastMath.sqrt(30.0)) / 180.0,
        (90.0 - 5.0 * FastMath.sqrt(30.0)) / 180.0
    };

    
    private static final double[] ABSCISSAS_5 = {
        -FastMath.sqrt((35.0 + 2.0 * FastMath.sqrt(70.0)) / 63.0),
        -FastMath.sqrt((35.0 - 2.0 * FastMath.sqrt(70.0)) / 63.0),
         0.0,
         FastMath.sqrt((35.0 - 2.0 * FastMath.sqrt(70.0)) / 63.0),
         FastMath.sqrt((35.0 + 2.0 * FastMath.sqrt(70.0)) / 63.0)
    };

    
    private static final double[] WEIGHTS_5 = {
        (322.0 - 13.0 * FastMath.sqrt(70.0)) / 900.0,
        (322.0 + 13.0 * FastMath.sqrt(70.0)) / 900.0,
        128.0 / 225.0,
        (322.0 + 13.0 * FastMath.sqrt(70.0)) / 900.0,
        (322.0 - 13.0 * FastMath.sqrt(70.0)) / 900.0
    };

    
    private final double[] abscissas;

    
    private final double[] weights;

    
    public LegendreGaussIntegrator(final int n,
                                   final double relativeAccuracy,
                                   final double absoluteAccuracy,
                                   final int minimalIterationCount,
                                   final int maximalIterationCount)
        throws MathIllegalArgumentException, NotStrictlyPositiveException, NumberIsTooSmallException {
        super(relativeAccuracy, absoluteAccuracy, minimalIterationCount, maximalIterationCount);
        switch(n) {
        case 2 :
            abscissas = ABSCISSAS_2;
            weights   = WEIGHTS_2;
            break;
        case 3 :
            abscissas = ABSCISSAS_3;
            weights   = WEIGHTS_3;
            break;
        case 4 :
            abscissas = ABSCISSAS_4;
            weights   = WEIGHTS_4;
            break;
        case 5 :
            abscissas = ABSCISSAS_5;
            weights   = WEIGHTS_5;
            break;
        default :
            throw new MathIllegalArgumentException(
                    LocalizedFormats.N_POINTS_GAUSS_LEGENDRE_INTEGRATOR_NOT_SUPPORTED,
                    n, 2, 5);
        }

    }

    
    public LegendreGaussIntegrator(final int n,
                                   final double relativeAccuracy,
                                   final double absoluteAccuracy)
        throws MathIllegalArgumentException {
        this(n, relativeAccuracy, absoluteAccuracy,
             DEFAULT_MIN_ITERATIONS_COUNT, DEFAULT_MAX_ITERATIONS_COUNT);
    }

    
    public LegendreGaussIntegrator(final int n,
                                   final int minimalIterationCount,
                                   final int maximalIterationCount)
        throws MathIllegalArgumentException {
        this(n, DEFAULT_RELATIVE_ACCURACY, DEFAULT_ABSOLUTE_ACCURACY,
             minimalIterationCount, maximalIterationCount);
    }

    
    @Override
    protected double doIntegrate()
        throws MathIllegalArgumentException, TooManyEvaluationsException, MaxCountExceededException {

        // compute first estimate with a single step
        double oldt = stage(1);

        int n = 2;
        while (true) {

            // improve integral with a larger number of steps
            final double t = stage(n);

            // estimate error
            final double delta = FastMath.abs(t - oldt);
            final double limit =
                FastMath.max(getAbsoluteAccuracy(),
                             getRelativeAccuracy() * (FastMath.abs(oldt) + FastMath.abs(t)) * 0.5);

            // check convergence
            if ((getIterations() + 1 >= getMinimalIterationCount()) && (delta <= limit)) {
                return t;
            }

            // prepare next iteration
            double ratio = FastMath.min(4, FastMath.pow(delta / limit, 0.5 / abscissas.length));
            n = FastMath.max((int) (ratio * n), n + 1);
            oldt = t;
            incrementCount();

        }

    }

    
    private double stage(final int n)
        throws TooManyEvaluationsException {

        // set up the step for the current stage
        final double step     = (getMax() - getMin()) / n;
        final double halfStep = step / 2.0;

        // integrate over all elementary steps
        double midPoint = getMin() + halfStep;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < abscissas.length; ++j) {
                sum += weights[j] * computeObjectiveValue(midPoint + halfStep * abscissas[j]);
            }
            midPoint += step;
        }

        return halfStep * sum;

    }

}
