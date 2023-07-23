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
package org.apache.lucene.util.hnsw.math.fitting.leastsquares;

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem.Evaluation;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.util.Incrementor;
import org.apache.lucene.util.hnsw.math.util.Precision;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class LevenbergMarquardtOptimizer implements LeastSquaresOptimizer {

    
    private static final double TWO_EPS = 2 * Precision.EPSILON;

    /* configuration parameters */
    
    private final double initialStepBoundFactor;
    
    private final double costRelativeTolerance;
    
    private final double parRelativeTolerance;
    
    private final double orthoTolerance;
    
    private final double qrRankingThreshold;

    
    public LevenbergMarquardtOptimizer() {
        this(100, 1e-10, 1e-10, 1e-10, Precision.SAFE_MIN);
    }

    
    public LevenbergMarquardtOptimizer(
            final double initialStepBoundFactor,
            final double costRelativeTolerance,
            final double parRelativeTolerance,
            final double orthoTolerance,
            final double qrRankingThreshold) {
        this.initialStepBoundFactor = initialStepBoundFactor;
        this.costRelativeTolerance = costRelativeTolerance;
        this.parRelativeTolerance = parRelativeTolerance;
        this.orthoTolerance = orthoTolerance;
        this.qrRankingThreshold = qrRankingThreshold;
    }

    
    public LevenbergMarquardtOptimizer withInitialStepBoundFactor(double newInitialStepBoundFactor) {
        return new LevenbergMarquardtOptimizer(
                newInitialStepBoundFactor,
                costRelativeTolerance,
                parRelativeTolerance,
                orthoTolerance,
                qrRankingThreshold);
    }

    
    public LevenbergMarquardtOptimizer withCostRelativeTolerance(double newCostRelativeTolerance) {
        return new LevenbergMarquardtOptimizer(
                initialStepBoundFactor,
                newCostRelativeTolerance,
                parRelativeTolerance,
                orthoTolerance,
                qrRankingThreshold);
    }

    
    public LevenbergMarquardtOptimizer withParameterRelativeTolerance(double newParRelativeTolerance) {
        return new LevenbergMarquardtOptimizer(
                initialStepBoundFactor,
                costRelativeTolerance,
                newParRelativeTolerance,
                orthoTolerance,
                qrRankingThreshold);
    }

    
    public LevenbergMarquardtOptimizer withOrthoTolerance(double newOrthoTolerance) {
        return new LevenbergMarquardtOptimizer(
                initialStepBoundFactor,
                costRelativeTolerance,
                parRelativeTolerance,
                newOrthoTolerance,
                qrRankingThreshold);
    }

    
    public LevenbergMarquardtOptimizer withRankingThreshold(double newQRRankingThreshold) {
        return new LevenbergMarquardtOptimizer(
                initialStepBoundFactor,
                costRelativeTolerance,
                parRelativeTolerance,
                orthoTolerance,
                newQRRankingThreshold);
    }

    
    public double getInitialStepBoundFactor() {
        return initialStepBoundFactor;
    }

    
    public double getCostRelativeTolerance() {
        return costRelativeTolerance;
    }

    
    public double getParameterRelativeTolerance() {
        return parRelativeTolerance;
    }

    
    public double getOrthoTolerance() {
        return orthoTolerance;
    }

    
    public double getRankingThreshold() {
        return qrRankingThreshold;
    }

    
    public Optimum optimize(final LeastSquaresProblem problem) {
        // Pull in relevant data from the problem as locals.
        final int nR = problem.getObservationSize(); // Number of observed data.
        final int nC = problem.getParameterSize(); // Number of parameters.
        // Counters.
        final Incrementor iterationCounter = problem.getIterationCounter();
        final Incrementor evaluationCounter = problem.getEvaluationCounter();
        // Convergence criterion.
        final ConvergenceChecker<Evaluation> checker = problem.getConvergenceChecker();

        // arrays shared with the other private methods
        final int solvedCols  = FastMath.min(nR, nC);
        /* Parameters evolution direction associated with lmPar. */
        double[] lmDir = new double[nC];
        /* Levenberg-Marquardt parameter. */
        double lmPar = 0;

        // local point
        double   delta   = 0;
        double   xNorm   = 0;
        double[] diag    = new double[nC];
        double[] oldX    = new double[nC];
        double[] oldRes  = new double[nR];
        double[] qtf     = new double[nR];
        double[] work1   = new double[nC];
        double[] work2   = new double[nC];
        double[] work3   = new double[nC];


        // Evaluate the function at the starting point and calculate its norm.
        evaluationCounter.incrementCount();
        //value will be reassigned in the loop
        Evaluation current = problem.evaluate(problem.getStart());
        double[] currentResiduals = current.getResiduals().toArray();
        double currentCost = current.getCost();
        double[] currentPoint = current.getPoint().toArray();

        // Outer loop.
        boolean firstIteration = true;
        while (true) {
            iterationCounter.incrementCount();

            final Evaluation previous = current;

            // QR decomposition of the jacobian matrix
            final InternalData internalData
                    = qrDecomposition(current.getJacobian(), solvedCols);
            final double[][] weightedJacobian = internalData.weightedJacobian;
            final int[] permutation = internalData.permutation;
            final double[] diagR = internalData.diagR;
            final double[] jacNorm = internalData.jacNorm;

            //residuals already have weights applied
            double[] weightedResidual = currentResiduals;
            for (int i = 0; i < nR; i++) {
                qtf[i] = weightedResidual[i];
            }

            // compute Qt.res
            qTy(qtf, internalData);

            // now we don't need Q anymore,
            // so let jacobian contain the R matrix with its diagonal elements
            for (int k = 0; k < solvedCols; ++k) {
                int pk = permutation[k];
                weightedJacobian[k][pk] = diagR[pk];
            }

            if (firstIteration) {
                // scale the point according to the norms of the columns
                // of the initial jacobian
                xNorm = 0;
                for (int k = 0; k < nC; ++k) {
                    double dk = jacNorm[k];
                    if (dk == 0) {
                        dk = 1.0;
                    }
                    double xk = dk * currentPoint[k];
                    xNorm  += xk * xk;
                    diag[k] = dk;
                }
                xNorm = FastMath.sqrt(xNorm);

                // initialize the step bound delta
                delta = (xNorm == 0) ? initialStepBoundFactor : (initialStepBoundFactor * xNorm);
            }

            // check orthogonality between function vector and jacobian columns
            double maxCosine = 0;
            if (currentCost != 0) {
                for (int j = 0; j < solvedCols; ++j) {
                    int    pj = permutation[j];
                    double s  = jacNorm[pj];
                    if (s != 0) {
                        double sum = 0;
                        for (int i = 0; i <= j; ++i) {
                            sum += weightedJacobian[i][pj] * qtf[i];
                        }
                        maxCosine = FastMath.max(maxCosine, FastMath.abs(sum) / (s * currentCost));
                    }
                }
            }
            if (maxCosine <= orthoTolerance) {
                // Convergence has been reached.
                return new OptimumImpl(
                        current,
                        evaluationCounter.getCount(),
                        iterationCounter.getCount());
            }

            // rescale if necessary
            for (int j = 0; j < nC; ++j) {
                diag[j] = FastMath.max(diag[j], jacNorm[j]);
            }

            // Inner loop.
            for (double ratio = 0; ratio < 1.0e-4;) {

                // save the state
                for (int j = 0; j < solvedCols; ++j) {
                    int pj = permutation[j];
                    oldX[pj] = currentPoint[pj];
                }
                final double previousCost = currentCost;
                double[] tmpVec = weightedResidual;
                weightedResidual = oldRes;
                oldRes    = tmpVec;

                // determine the Levenberg-Marquardt parameter
                lmPar = determineLMParameter(qtf, delta, diag,
                                     internalData, solvedCols,
                                     work1, work2, work3, lmDir, lmPar);

                // compute the new point and the norm of the evolution direction
                double lmNorm = 0;
                for (int j = 0; j < solvedCols; ++j) {
                    int pj = permutation[j];
                    lmDir[pj] = -lmDir[pj];
                    currentPoint[pj] = oldX[pj] + lmDir[pj];
                    double s = diag[pj] * lmDir[pj];
                    lmNorm  += s * s;
                }
                lmNorm = FastMath.sqrt(lmNorm);
                // on the first iteration, adjust the initial step bound.
                if (firstIteration) {
                    delta = FastMath.min(delta, lmNorm);
                }

                // Evaluate the function at x + p and calculate its norm.
                evaluationCounter.incrementCount();
                current = problem.evaluate(new ArrayRealVector(currentPoint));
                currentResiduals = current.getResiduals().toArray();
                currentCost = current.getCost();
                currentPoint = current.getPoint().toArray();

                // compute the scaled actual reduction
                double actRed = -1.0;
                if (0.1 * currentCost < previousCost) {
                    double r = currentCost / previousCost;
                    actRed = 1.0 - r * r;
                }

                // compute the scaled predicted reduction
                // and the scaled directional derivative
                for (int j = 0; j < solvedCols; ++j) {
                    int pj = permutation[j];
                    double dirJ = lmDir[pj];
                    work1[j] = 0;
                    for (int i = 0; i <= j; ++i) {
                        work1[i] += weightedJacobian[i][pj] * dirJ;
                    }
                }
                double coeff1 = 0;
                for (int j = 0; j < solvedCols; ++j) {
                    coeff1 += work1[j] * work1[j];
                }
                double pc2 = previousCost * previousCost;
                coeff1 /= pc2;
                double coeff2 = lmPar * lmNorm * lmNorm / pc2;
                double preRed = coeff1 + 2 * coeff2;
                double dirDer = -(coeff1 + coeff2);

                // ratio of the actual to the predicted reduction
                ratio = (preRed == 0) ? 0 : (actRed / preRed);

                // update the step bound
                if (ratio <= 0.25) {
                    double tmp =
                        (actRed < 0) ? (0.5 * dirDer / (dirDer + 0.5 * actRed)) : 0.5;
                        if ((0.1 * currentCost >= previousCost) || (tmp < 0.1)) {
                            tmp = 0.1;
                        }
                        delta = tmp * FastMath.min(delta, 10.0 * lmNorm);
                        lmPar /= tmp;
                } else if ((lmPar == 0) || (ratio >= 0.75)) {
                    delta = 2 * lmNorm;
                    lmPar *= 0.5;
                }

                // test for successful iteration.
                if (ratio >= 1.0e-4) {
                    // successful iteration, update the norm
                    firstIteration = false;
                    xNorm = 0;
                    for (int k = 0; k < nC; ++k) {
                        double xK = diag[k] * currentPoint[k];
                        xNorm += xK * xK;
                    }
                    xNorm = FastMath.sqrt(xNorm);

                    // tests for convergence.
                    if (checker != null && checker.converged(iterationCounter.getCount(), previous, current)) {
                        return new OptimumImpl(current, evaluationCounter.getCount(), iterationCounter.getCount());
                    }
                } else {
                    // failed iteration, reset the previous values
                    currentCost = previousCost;
                    for (int j = 0; j < solvedCols; ++j) {
                        int pj = permutation[j];
                        currentPoint[pj] = oldX[pj];
                    }
                    tmpVec    = weightedResidual;
                    weightedResidual = oldRes;
                    oldRes    = tmpVec;
                    // Reset "current" to previous values.
                    current = previous;
                }

                // Default convergence criteria.
                if ((FastMath.abs(actRed) <= costRelativeTolerance &&
                     preRed <= costRelativeTolerance &&
                     ratio <= 2.0) ||
                    delta <= parRelativeTolerance * xNorm) {
                    return new OptimumImpl(current, evaluationCounter.getCount(), iterationCounter.getCount());
                }

                // tests for termination and stringent tolerances
                if (FastMath.abs(actRed) <= TWO_EPS &&
                    preRed <= TWO_EPS &&
                    ratio <= 2.0) {
                    throw new ConvergenceException(LocalizedFormats.TOO_SMALL_COST_RELATIVE_TOLERANCE,
                                                   costRelativeTolerance);
                } else if (delta <= TWO_EPS * xNorm) {
                    throw new ConvergenceException(LocalizedFormats.TOO_SMALL_PARAMETERS_RELATIVE_TOLERANCE,
                                                   parRelativeTolerance);
                } else if (maxCosine <= TWO_EPS) {
                    throw new ConvergenceException(LocalizedFormats.TOO_SMALL_ORTHOGONALITY_TOLERANCE,
                                                   orthoTolerance);
                }
            }
        }
    }

    
    private static class InternalData {
        
        private final double[][] weightedJacobian;
        
        private final int[] permutation;
        
        private final int rank;
        
        private final double[] diagR;
        
        private final double[] jacNorm;
        
        private final double[] beta;

        
        InternalData(double[][] weightedJacobian,
                     int[] permutation,
                     int rank,
                     double[] diagR,
                     double[] jacNorm,
                     double[] beta) {
            this.weightedJacobian = weightedJacobian;
            this.permutation = permutation;
            this.rank = rank;
            this.diagR = diagR;
            this.jacNorm = jacNorm;
            this.beta = beta;
        }
    }

    
    private double determineLMParameter(double[] qy, double delta, double[] diag,
                                      InternalData internalData, int solvedCols,
                                      double[] work1, double[] work2, double[] work3,
                                      double[] lmDir, double lmPar) {
        final double[][] weightedJacobian = internalData.weightedJacobian;
        final int[] permutation = internalData.permutation;
        final int rank = internalData.rank;
        final double[] diagR = internalData.diagR;

        final int nC = weightedJacobian[0].length;

        // compute and store in x the gauss-newton direction, if the
        // jacobian is rank-deficient, obtain a least squares solution
        for (int j = 0; j < rank; ++j) {
            lmDir[permutation[j]] = qy[j];
        }
        for (int j = rank; j < nC; ++j) {
            lmDir[permutation[j]] = 0;
        }
        for (int k = rank - 1; k >= 0; --k) {
            int pk = permutation[k];
            double ypk = lmDir[pk] / diagR[pk];
            for (int i = 0; i < k; ++i) {
                lmDir[permutation[i]] -= ypk * weightedJacobian[i][pk];
            }
            lmDir[pk] = ypk;
        }

        // evaluate the function at the origin, and test
        // for acceptance of the Gauss-Newton direction
        double dxNorm = 0;
        for (int j = 0; j < solvedCols; ++j) {
            int pj = permutation[j];
            double s = diag[pj] * lmDir[pj];
            work1[pj] = s;
            dxNorm += s * s;
        }
        dxNorm = FastMath.sqrt(dxNorm);
        double fp = dxNorm - delta;
        if (fp <= 0.1 * delta) {
            lmPar = 0;
            return lmPar;
        }

        // if the jacobian is not rank deficient, the Newton step provides
        // a lower bound, parl, for the zero of the function,
        // otherwise set this bound to zero
        double sum2;
        double parl = 0;
        if (rank == solvedCols) {
            for (int j = 0; j < solvedCols; ++j) {
                int pj = permutation[j];
                work1[pj] *= diag[pj] / dxNorm;
            }
            sum2 = 0;
            for (int j = 0; j < solvedCols; ++j) {
                int pj = permutation[j];
                double sum = 0;
                for (int i = 0; i < j; ++i) {
                    sum += weightedJacobian[i][pj] * work1[permutation[i]];
                }
                double s = (work1[pj] - sum) / diagR[pj];
                work1[pj] = s;
                sum2 += s * s;
            }
            parl = fp / (delta * sum2);
        }

        // calculate an upper bound, paru, for the zero of the function
        sum2 = 0;
        for (int j = 0; j < solvedCols; ++j) {
            int pj = permutation[j];
            double sum = 0;
            for (int i = 0; i <= j; ++i) {
                sum += weightedJacobian[i][pj] * qy[i];
            }
            sum /= diag[pj];
            sum2 += sum * sum;
        }
        double gNorm = FastMath.sqrt(sum2);
        double paru = gNorm / delta;
        if (paru == 0) {
            paru = Precision.SAFE_MIN / FastMath.min(delta, 0.1);
        }

        // if the input par lies outside of the interval (parl,paru),
        // set par to the closer endpoint
        lmPar = FastMath.min(paru, FastMath.max(lmPar, parl));
        if (lmPar == 0) {
            lmPar = gNorm / dxNorm;
        }

        for (int countdown = 10; countdown >= 0; --countdown) {

            // evaluate the function at the current value of lmPar
            if (lmPar == 0) {
                lmPar = FastMath.max(Precision.SAFE_MIN, 0.001 * paru);
            }
            double sPar = FastMath.sqrt(lmPar);
            for (int j = 0; j < solvedCols; ++j) {
                int pj = permutation[j];
                work1[pj] = sPar * diag[pj];
            }
            determineLMDirection(qy, work1, work2, internalData, solvedCols, work3, lmDir);

            dxNorm = 0;
            for (int j = 0; j < solvedCols; ++j) {
                int pj = permutation[j];
                double s = diag[pj] * lmDir[pj];
                work3[pj] = s;
                dxNorm += s * s;
            }
            dxNorm = FastMath.sqrt(dxNorm);
            double previousFP = fp;
            fp = dxNorm - delta;

            // if the function is small enough, accept the current value
            // of lmPar, also test for the exceptional cases where parl is zero
            if (FastMath.abs(fp) <= 0.1 * delta ||
                (parl == 0 &&
                 fp <= previousFP &&
                 previousFP < 0)) {
                return lmPar;
            }

            // compute the Newton correction
            for (int j = 0; j < solvedCols; ++j) {
                int pj = permutation[j];
                work1[pj] = work3[pj] * diag[pj] / dxNorm;
            }
            for (int j = 0; j < solvedCols; ++j) {
                int pj = permutation[j];
                work1[pj] /= work2[j];
                double tmp = work1[pj];
                for (int i = j + 1; i < solvedCols; ++i) {
                    work1[permutation[i]] -= weightedJacobian[i][pj] * tmp;
                }
            }
            sum2 = 0;
            for (int j = 0; j < solvedCols; ++j) {
                double s = work1[permutation[j]];
                sum2 += s * s;
            }
            double correction = fp / (delta * sum2);

            // depending on the sign of the function, update parl or paru.
            if (fp > 0) {
                parl = FastMath.max(parl, lmPar);
            } else if (fp < 0) {
                paru = FastMath.min(paru, lmPar);
            }

            // compute an improved estimate for lmPar
            lmPar = FastMath.max(parl, lmPar + correction);
        }

        return lmPar;
    }

    
    private void determineLMDirection(double[] qy, double[] diag,
                                      double[] lmDiag,
                                      InternalData internalData,
                                      int solvedCols,
                                      double[] work,
                                      double[] lmDir) {
        final int[] permutation = internalData.permutation;
        final double[][] weightedJacobian = internalData.weightedJacobian;
        final double[] diagR = internalData.diagR;

        // copy R and Qty to preserve input and initialize s
        //  in particular, save the diagonal elements of R in lmDir
        for (int j = 0; j < solvedCols; ++j) {
            int pj = permutation[j];
            for (int i = j + 1; i < solvedCols; ++i) {
                weightedJacobian[i][pj] = weightedJacobian[j][permutation[i]];
            }
            lmDir[j] = diagR[pj];
            work[j]  = qy[j];
        }

        // eliminate the diagonal matrix d using a Givens rotation
        for (int j = 0; j < solvedCols; ++j) {

            // prepare the row of d to be eliminated, locating the
            // diagonal element using p from the Q.R. factorization
            int pj = permutation[j];
            double dpj = diag[pj];
            if (dpj != 0) {
                Arrays.fill(lmDiag, j + 1, lmDiag.length, 0);
            }
            lmDiag[j] = dpj;

            //  the transformations to eliminate the row of d
            // modify only a single element of Qty
            // beyond the first n, which is initially zero.
            double qtbpj = 0;
            for (int k = j; k < solvedCols; ++k) {
                int pk = permutation[k];

                // determine a Givens rotation which eliminates the
                // appropriate element in the current row of d
                if (lmDiag[k] != 0) {

                    final double sin;
                    final double cos;
                    double rkk = weightedJacobian[k][pk];
                    if (FastMath.abs(rkk) < FastMath.abs(lmDiag[k])) {
                        final double cotan = rkk / lmDiag[k];
                        sin   = 1.0 / FastMath.sqrt(1.0 + cotan * cotan);
                        cos   = sin * cotan;
                    } else {
                        final double tan = lmDiag[k] / rkk;
                        cos = 1.0 / FastMath.sqrt(1.0 + tan * tan);
                        sin = cos * tan;
                    }

                    // compute the modified diagonal element of R and
                    // the modified element of (Qty,0)
                    weightedJacobian[k][pk] = cos * rkk + sin * lmDiag[k];
                    final double temp = cos * work[k] + sin * qtbpj;
                    qtbpj = -sin * work[k] + cos * qtbpj;
                    work[k] = temp;

                    // accumulate the tranformation in the row of s
                    for (int i = k + 1; i < solvedCols; ++i) {
                        double rik = weightedJacobian[i][pk];
                        final double temp2 = cos * rik + sin * lmDiag[i];
                        lmDiag[i] = -sin * rik + cos * lmDiag[i];
                        weightedJacobian[i][pk] = temp2;
                    }
                }
            }

            // store the diagonal element of s and restore
            // the corresponding diagonal element of R
            lmDiag[j] = weightedJacobian[j][permutation[j]];
            weightedJacobian[j][permutation[j]] = lmDir[j];
        }

        // solve the triangular system for z, if the system is
        // singular, then obtain a least squares solution
        int nSing = solvedCols;
        for (int j = 0; j < solvedCols; ++j) {
            if ((lmDiag[j] == 0) && (nSing == solvedCols)) {
                nSing = j;
            }
            if (nSing < solvedCols) {
                work[j] = 0;
            }
        }
        if (nSing > 0) {
            for (int j = nSing - 1; j >= 0; --j) {
                int pj = permutation[j];
                double sum = 0;
                for (int i = j + 1; i < nSing; ++i) {
                    sum += weightedJacobian[i][pj] * work[i];
                }
                work[j] = (work[j] - sum) / lmDiag[j];
            }
        }

        // permute the components of z back to components of lmDir
        for (int j = 0; j < lmDir.length; ++j) {
            lmDir[permutation[j]] = work[j];
        }
    }

    
    private InternalData qrDecomposition(RealMatrix jacobian,
                                         int solvedCols) throws ConvergenceException {
        // Code in this class assumes that the weighted Jacobian is -(W^(1/2) J),
        // hence the multiplication by -1.
        final double[][] weightedJacobian = jacobian.scalarMultiply(-1).getData();

        final int nR = weightedJacobian.length;
        final int nC = weightedJacobian[0].length;

        final int[] permutation = new int[nC];
        final double[] diagR = new double[nC];
        final double[] jacNorm = new double[nC];
        final double[] beta = new double[nC];

        // initializations
        for (int k = 0; k < nC; ++k) {
            permutation[k] = k;
            double norm2 = 0;
            for (int i = 0; i < nR; ++i) {
                double akk = weightedJacobian[i][k];
                norm2 += akk * akk;
            }
            jacNorm[k] = FastMath.sqrt(norm2);
        }

        // transform the matrix column after column
        for (int k = 0; k < nC; ++k) {

            // select the column with the greatest norm on active components
            int nextColumn = -1;
            double ak2 = Double.NEGATIVE_INFINITY;
            for (int i = k; i < nC; ++i) {
                double norm2 = 0;
                for (int j = k; j < nR; ++j) {
                    double aki = weightedJacobian[j][permutation[i]];
                    norm2 += aki * aki;
                }
                if (Double.isInfinite(norm2) || Double.isNaN(norm2)) {
                    throw new ConvergenceException(LocalizedFormats.UNABLE_TO_PERFORM_QR_DECOMPOSITION_ON_JACOBIAN,
                                                   nR, nC);
                }
                if (norm2 > ak2) {
                    nextColumn = i;
                    ak2        = norm2;
                }
            }
            if (ak2 <= qrRankingThreshold) {
                return new InternalData(weightedJacobian, permutation, k, diagR, jacNorm, beta);
            }
            int pk = permutation[nextColumn];
            permutation[nextColumn] = permutation[k];
            permutation[k] = pk;

            // choose alpha such that Hk.u = alpha ek
            double akk = weightedJacobian[k][pk];
            double alpha = (akk > 0) ? -FastMath.sqrt(ak2) : FastMath.sqrt(ak2);
            double betak = 1.0 / (ak2 - akk * alpha);
            beta[pk] = betak;

            // transform the current column
            diagR[pk] = alpha;
            weightedJacobian[k][pk] -= alpha;

            // transform the remaining columns
            for (int dk = nC - 1 - k; dk > 0; --dk) {
                double gamma = 0;
                for (int j = k; j < nR; ++j) {
                    gamma += weightedJacobian[j][pk] * weightedJacobian[j][permutation[k + dk]];
                }
                gamma *= betak;
                for (int j = k; j < nR; ++j) {
                    weightedJacobian[j][permutation[k + dk]] -= gamma * weightedJacobian[j][pk];
                }
            }
        }

        return new InternalData(weightedJacobian, permutation, solvedCols, diagR, jacNorm, beta);
    }

    
    private void qTy(double[] y,
                     InternalData internalData) {
        final double[][] weightedJacobian = internalData.weightedJacobian;
        final int[] permutation = internalData.permutation;
        final double[] beta = internalData.beta;

        final int nR = weightedJacobian.length;
        final int nC = weightedJacobian[0].length;

        for (int k = 0; k < nC; ++k) {
            int pk = permutation[k];
            double gamma = 0;
            for (int i = k; i < nR; ++i) {
                gamma += weightedJacobian[i][pk] * y[i];
            }
            gamma *= beta[pk];
            for (int i = k; i < nR; ++i) {
                y[i] -= gamma * weightedJacobian[i][pk];
            }
        }
    }
}
