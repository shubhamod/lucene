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
package org.apache.lucene.util.hnsw.math.fitting;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.lucene.util.hnsw.math.analysis.function.HarmonicOscillator;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem;
import org.apache.lucene.util.hnsw.math.linear.DiagonalMatrix;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class HarmonicCurveFitter extends AbstractCurveFitter {
    
    private static final HarmonicOscillator.Parametric FUNCTION = new HarmonicOscillator.Parametric();
    
    private final double[] initialGuess;
    
    private final int maxIter;

    
    private HarmonicCurveFitter(double[] initialGuess,
                                int maxIter) {
        this.initialGuess = initialGuess;
        this.maxIter = maxIter;
    }

    
    public static HarmonicCurveFitter create() {
        return new HarmonicCurveFitter(null, Integer.MAX_VALUE);
    }

    
    public HarmonicCurveFitter withStartPoint(double[] newStart) {
        return new HarmonicCurveFitter(newStart.clone(),
                                       maxIter);
    }

    
    public HarmonicCurveFitter withMaxIterations(int newMaxIter) {
        return new HarmonicCurveFitter(initialGuess,
                                       newMaxIter);
    }

    
    @Override
    protected LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> observations) {
        // Prepare least-squares problem.
        final int len = observations.size();
        final double[] target  = new double[len];
        final double[] weights = new double[len];

        int i = 0;
        for (WeightedObservedPoint obs : observations) {
            target[i]  = obs.getY();
            weights[i] = obs.getWeight();
            ++i;
        }

        final AbstractCurveFitter.TheoreticalValuesFunction model
            = new AbstractCurveFitter.TheoreticalValuesFunction(FUNCTION,
                                                                observations);

        final double[] startPoint = initialGuess != null ?
            initialGuess :
            // Compute estimation.
            new ParameterGuesser(observations).guess();

        // Return a new optimizer set up to fit a Gaussian curve to the
        // observed points.
        return new LeastSquaresBuilder().
                maxEvaluations(Integer.MAX_VALUE).
                maxIterations(maxIter).
                start(startPoint).
                target(target).
                weight(new DiagonalMatrix(weights)).
                model(model.getModelFunction(), model.getModelFunctionJacobian()).
                build();

    }

    
    public static class ParameterGuesser {
        
        private final double a;
        
        private final double omega;
        
        private final double phi;

        
        public ParameterGuesser(Collection<WeightedObservedPoint> observations) {
            if (observations.size() < 4) {
                throw new NumberIsTooSmallException(LocalizedFormats.INSUFFICIENT_OBSERVED_POINTS_IN_SAMPLE,
                                                    observations.size(), 4, true);
            }

            final WeightedObservedPoint[] sorted
                = sortObservations(observations).toArray(new WeightedObservedPoint[0]);

            final double aOmega[] = guessAOmega(sorted);
            a = aOmega[0];
            omega = aOmega[1];

            phi = guessPhi(sorted);
        }

        
        public double[] guess() {
            return new double[] { a, omega, phi };
        }

        
        private List<WeightedObservedPoint> sortObservations(Collection<WeightedObservedPoint> unsorted) {
            final List<WeightedObservedPoint> observations = new ArrayList<WeightedObservedPoint>(unsorted);

            // Since the samples are almost always already sorted, this
            // method is implemented as an insertion sort that reorders the
            // elements in place. Insertion sort is very efficient in this case.
            WeightedObservedPoint curr = observations.get(0);
            final int len = observations.size();
            for (int j = 1; j < len; j++) {
                WeightedObservedPoint prec = curr;
                curr = observations.get(j);
                if (curr.getX() < prec.getX()) {
                    // the current element should be inserted closer to the beginning
                    int i = j - 1;
                    WeightedObservedPoint mI = observations.get(i);
                    while ((i >= 0) && (curr.getX() < mI.getX())) {
                        observations.set(i + 1, mI);
                        if (i-- != 0) {
                            mI = observations.get(i);
                        }
                    }
                    observations.set(i + 1, curr);
                    curr = observations.get(j);
                }
            }

            return observations;
        }

        
        private double[] guessAOmega(WeightedObservedPoint[] observations) {
            final double[] aOmega = new double[2];

            // initialize the sums for the linear model between the two integrals
            double sx2 = 0;
            double sy2 = 0;
            double sxy = 0;
            double sxz = 0;
            double syz = 0;

            double currentX = observations[0].getX();
            double currentY = observations[0].getY();
            double f2Integral = 0;
            double fPrime2Integral = 0;
            final double startX = currentX;
            for (int i = 1; i < observations.length; ++i) {
                // one step forward
                final double previousX = currentX;
                final double previousY = currentY;
                currentX = observations[i].getX();
                currentY = observations[i].getY();

                // update the integrals of f<sup>2</sup> and f'<sup>2</sup>
                // considering a linear model for f (and therefore constant f')
                final double dx = currentX - previousX;
                final double dy = currentY - previousY;
                final double f2StepIntegral =
                    dx * (previousY * previousY + previousY * currentY + currentY * currentY) / 3;
                final double fPrime2StepIntegral = dy * dy / dx;

                final double x = currentX - startX;
                f2Integral += f2StepIntegral;
                fPrime2Integral += fPrime2StepIntegral;

                sx2 += x * x;
                sy2 += f2Integral * f2Integral;
                sxy += x * f2Integral;
                sxz += x * fPrime2Integral;
                syz += f2Integral * fPrime2Integral;
            }

            // compute the amplitude and pulsation coefficients
            double c1 = sy2 * sxz - sxy * syz;
            double c2 = sxy * sxz - sx2 * syz;
            double c3 = sx2 * sy2 - sxy * sxy;
            if ((c1 / c2 < 0) || (c2 / c3 < 0)) {
                final int last = observations.length - 1;
                // Range of the observations, assuming that the
                // observations are sorted.
                final double xRange = observations[last].getX() - observations[0].getX();
                if (xRange == 0) {
                    throw new ZeroException();
                }
                aOmega[1] = 2 * Math.PI / xRange;

                double yMin = Double.POSITIVE_INFINITY;
                double yMax = Double.NEGATIVE_INFINITY;
                for (int i = 1; i < observations.length; ++i) {
                    final double y = observations[i].getY();
                    if (y < yMin) {
                        yMin = y;
                    }
                    if (y > yMax) {
                        yMax = y;
                    }
                }
                aOmega[0] = 0.5 * (yMax - yMin);
            } else {
                if (c2 == 0) {
                    // In some ill-conditioned cases (cf. MATH-844), the guesser
                    // procedure cannot produce sensible results.
                    throw new MathIllegalStateException(LocalizedFormats.ZERO_DENOMINATOR);
                }

                aOmega[0] = FastMath.sqrt(c1 / c2);
                aOmega[1] = FastMath.sqrt(c2 / c3);
            }

            return aOmega;
        }

        
        private double guessPhi(WeightedObservedPoint[] observations) {
            // initialize the means
            double fcMean = 0;
            double fsMean = 0;

            double currentX = observations[0].getX();
            double currentY = observations[0].getY();
            for (int i = 1; i < observations.length; ++i) {
                // one step forward
                final double previousX = currentX;
                final double previousY = currentY;
                currentX = observations[i].getX();
                currentY = observations[i].getY();
                final double currentYPrime = (currentY - previousY) / (currentX - previousX);

                double omegaX = omega * currentX;
                double cosine = FastMath.cos(omegaX);
                double sine = FastMath.sin(omegaX);
                fcMean += omega * currentY * cosine - currentYPrime * sine;
                fsMean += omega * currentY * sine + currentYPrime * cosine;
            }

            return FastMath.atan2(-fsMean, fcMean);
        }
    }
}
