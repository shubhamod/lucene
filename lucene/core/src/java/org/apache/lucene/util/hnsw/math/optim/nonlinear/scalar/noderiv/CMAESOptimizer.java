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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.EigenDecomposition;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.optim.ConvergenceChecker;
import org.apache.lucene.util.hnsw.math.optim.OptimizationData;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.GoalType;
import org.apache.lucene.util.hnsw.math.optim.PointValuePair;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class CMAESOptimizer
    extends MultivariateOptimizer {
    // global search parameters
    
    private int lambda; // population size
    
    private final boolean isActiveCMA;
    
    private final int checkFeasableCount;
    
    private double[] inputSigma;
    
    private int dimension;
    
    private int diagonalOnly;
    
    private boolean isMinimize = true;
    
    private final boolean generateStatistics;

    // termination criteria
    
    private final int maxIterations;
    
    private final double stopFitness;
    
    private double stopTolUpX;
    
    private double stopTolX;
    
    private double stopTolFun;
    
    private double stopTolHistFun;

    // selection strategy parameters
    
    private int mu; //
    
    private double logMu2;
    
    private RealMatrix weights;
    
    private double mueff; //

    // dynamic strategy parameters and constants
    
    private double sigma;
    
    private double cc;
    
    private double cs;
    
    private double damps;
    
    private double ccov1;
    
    private double ccovmu;
    
    private double chiN;
    
    private double ccov1Sep;
    
    private double ccovmuSep;

    // CMA internal values - updated each generation
    
    private RealMatrix xmean;
    
    private RealMatrix pc;
    
    private RealMatrix ps;
    
    private double normps;
    
    private RealMatrix B;
    
    private RealMatrix D;
    
    private RealMatrix BD;
    
    private RealMatrix diagD;
    
    private RealMatrix C;
    
    private RealMatrix diagC;
    
    private int iterations;

    
    private double[] fitnessHistory;
    
    private int historySize;

    
    private final RandomGenerator random;

    
    private final List<Double> statisticsSigmaHistory = new ArrayList<Double>();
    
    private final List<RealMatrix> statisticsMeanHistory = new ArrayList<RealMatrix>();
    
    private final List<Double> statisticsFitnessHistory = new ArrayList<Double>();
    
    private final List<RealMatrix> statisticsDHistory = new ArrayList<RealMatrix>();

    
    public CMAESOptimizer(int maxIterations,
                          double stopFitness,
                          boolean isActiveCMA,
                          int diagonalOnly,
                          int checkFeasableCount,
                          RandomGenerator random,
                          boolean generateStatistics,
                          ConvergenceChecker<PointValuePair> checker) {
        super(checker);
        this.maxIterations = maxIterations;
        this.stopFitness = stopFitness;
        this.isActiveCMA = isActiveCMA;
        this.diagonalOnly = diagonalOnly;
        this.checkFeasableCount = checkFeasableCount;
        this.random = random;
        this.generateStatistics = generateStatistics;
    }

    
    public List<Double> getStatisticsSigmaHistory() {
        return statisticsSigmaHistory;
    }

    
    public List<RealMatrix> getStatisticsMeanHistory() {
        return statisticsMeanHistory;
    }

    
    public List<Double> getStatisticsFitnessHistory() {
        return statisticsFitnessHistory;
    }

    
    public List<RealMatrix> getStatisticsDHistory() {
        return statisticsDHistory;
    }

    
    public static class Sigma implements OptimizationData {
        
        private final double[] sigma;

        
        public Sigma(double[] s)
            throws NotPositiveException {
            for (int i = 0; i < s.length; i++) {
                if (s[i] < 0) {
                    throw new NotPositiveException(s[i]);
                }
            }

            sigma = s.clone();
        }

        
        public double[] getSigma() {
            return sigma.clone();
        }
    }

    
    public static class PopulationSize implements OptimizationData {
        
        private final int lambda;

        
        public PopulationSize(int size)
            throws NotStrictlyPositiveException {
            if (size <= 0) {
                throw new NotStrictlyPositiveException(size);
            }
            lambda = size;
        }

        
        public int getPopulationSize() {
            return lambda;
        }
    }

    
    @Override
    public PointValuePair optimize(OptimizationData... optData)
        throws TooManyEvaluationsException,
               DimensionMismatchException {
        // Set up base class and perform computation.
        return super.optimize(optData);
    }

    
    @Override
    protected PointValuePair doOptimize() {
         // -------------------- Initialization --------------------------------
        isMinimize = getGoalType().equals(GoalType.MINIMIZE);
        final FitnessFunction fitfun = new FitnessFunction();
        final double[] guess = getStartPoint();
        // number of objective variables/problem dimension
        dimension = guess.length;
        initializeCMA(guess);
        iterations = 0;
        ValuePenaltyPair valuePenalty = fitfun.value(guess);
        double bestValue = valuePenalty.value+valuePenalty.penalty;
        push(fitnessHistory, bestValue);
        PointValuePair optimum
            = new PointValuePair(getStartPoint(),
                                 isMinimize ? bestValue : -bestValue);
        PointValuePair lastResult = null;

        // -------------------- Generation Loop --------------------------------

        generationLoop:
        for (iterations = 1; iterations <= maxIterations; iterations++) {
            incrementIterationCount();

            // Generate and evaluate lambda offspring
            final RealMatrix arz = randn1(dimension, lambda);
            final RealMatrix arx = zeros(dimension, lambda);
            final double[] fitness = new double[lambda];
            final ValuePenaltyPair[] valuePenaltyPairs = new ValuePenaltyPair[lambda];
            // generate random offspring
            for (int k = 0; k < lambda; k++) {
                RealMatrix arxk = null;
                for (int i = 0; i < checkFeasableCount + 1; i++) {
                    if (diagonalOnly <= 0) {
                        arxk = xmean.add(BD.multiply(arz.getColumnMatrix(k))
                                         .scalarMultiply(sigma)); // m + sig * Normal(0,C)
                    } else {
                        arxk = xmean.add(times(diagD,arz.getColumnMatrix(k))
                                         .scalarMultiply(sigma));
                    }
                    if (i >= checkFeasableCount ||
                        fitfun.isFeasible(arxk.getColumn(0))) {
                        break;
                    }
                    // regenerate random arguments for row
                    arz.setColumn(k, randn(dimension));
                }
                copyColumn(arxk, 0, arx, k);
                try {
                    valuePenaltyPairs[k] = fitfun.value(arx.getColumn(k)); // compute fitness
                } catch (TooManyEvaluationsException e) {
                    break generationLoop;
                }
            }

            // Compute fitnesses by adding value and penalty after scaling by value range.
            double valueRange = valueRange(valuePenaltyPairs);
            for (int iValue=0;iValue<valuePenaltyPairs.length;iValue++) {
                 fitness[iValue] = valuePenaltyPairs[iValue].value + valuePenaltyPairs[iValue].penalty*valueRange;
            }

            // Sort by fitness and compute weighted mean into xmean
            final int[] arindex = sortedIndices(fitness);
            // Calculate new xmean, this is selection and recombination
            final RealMatrix xold = xmean; // for speed up of Eq. (2) and (3)
            final RealMatrix bestArx = selectColumns(arx, MathArrays.copyOf(arindex, mu));
            xmean = bestArx.multiply(weights);
            final RealMatrix bestArz = selectColumns(arz, MathArrays.copyOf(arindex, mu));
            final RealMatrix zmean = bestArz.multiply(weights);
            final boolean hsig = updateEvolutionPaths(zmean, xold);
            if (diagonalOnly <= 0) {
                updateCovariance(hsig, bestArx, arz, arindex, xold);
            } else {
                updateCovarianceDiagonalOnly(hsig, bestArz);
            }
            // Adapt step size sigma - Eq. (5)
            sigma *= FastMath.exp(FastMath.min(1, (normps/chiN - 1) * cs / damps));
            final double bestFitness = fitness[arindex[0]];
            final double worstFitness = fitness[arindex[arindex.length - 1]];
            if (bestValue > bestFitness) {
                bestValue = bestFitness;
                lastResult = optimum;
                optimum = new PointValuePair(fitfun.repair(bestArx.getColumn(0)),
                                             isMinimize ? bestFitness : -bestFitness);
                if (getConvergenceChecker() != null && lastResult != null &&
                    getConvergenceChecker().converged(iterations, optimum, lastResult)) {
                    break generationLoop;
                }
            }
            // handle termination criteria
            // Break, if fitness is good enough
            if (stopFitness != 0 && bestFitness < (isMinimize ? stopFitness : -stopFitness)) {
                break generationLoop;
            }
            final double[] sqrtDiagC = sqrt(diagC).getColumn(0);
            final double[] pcCol = pc.getColumn(0);
            for (int i = 0; i < dimension; i++) {
                if (sigma * FastMath.max(FastMath.abs(pcCol[i]), sqrtDiagC[i]) > stopTolX) {
                    break;
                }
                if (i >= dimension - 1) {
                    break generationLoop;
                }
            }
            for (int i = 0; i < dimension; i++) {
                if (sigma * sqrtDiagC[i] > stopTolUpX) {
                    break generationLoop;
                }
            }
            final double historyBest = min(fitnessHistory);
            final double historyWorst = max(fitnessHistory);
            if (iterations > 2 &&
                FastMath.max(historyWorst, worstFitness) -
                FastMath.min(historyBest, bestFitness) < stopTolFun) {
                break generationLoop;
            }
            if (iterations > fitnessHistory.length &&
                historyWorst - historyBest < stopTolHistFun) {
                break generationLoop;
            }
            // condition number of the covariance matrix exceeds 1e14
            if (max(diagD) / min(diagD) > 1e7) {
                break generationLoop;
            }
            // user defined termination
            if (getConvergenceChecker() != null) {
                final PointValuePair current
                    = new PointValuePair(bestArx.getColumn(0),
                                         isMinimize ? bestFitness : -bestFitness);
                if (lastResult != null &&
                    getConvergenceChecker().converged(iterations, current, lastResult)) {
                    break generationLoop;
                    }
                lastResult = current;
            }
            // Adjust step size in case of equal function values (flat fitness)
            if (bestValue == fitness[arindex[(int)(0.1+lambda/4.)]]) {
                sigma *= FastMath.exp(0.2 + cs / damps);
            }
            if (iterations > 2 && FastMath.max(historyWorst, bestFitness) -
                FastMath.min(historyBest, bestFitness) == 0) {
                sigma *= FastMath.exp(0.2 + cs / damps);
            }
            // store best in history
            push(fitnessHistory,bestFitness);
            if (generateStatistics) {
                statisticsSigmaHistory.add(sigma);
                statisticsFitnessHistory.add(bestFitness);
                statisticsMeanHistory.add(xmean.transpose());
                statisticsDHistory.add(diagD.transpose().scalarMultiply(1E5));
            }
        }
        return optimum;
    }

    
    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        // Allow base class to register its own data.
        super.parseOptimizationData(optData);

        // The existing values (as set by the previous call) are reused if
        // not provided in the argument list.
        for (OptimizationData data : optData) {
            if (data instanceof Sigma) {
                inputSigma = ((Sigma) data).getSigma();
                continue;
            }
            if (data instanceof PopulationSize) {
                lambda = ((PopulationSize) data).getPopulationSize();
                continue;
            }
        }

        checkParameters();
    }

    
    private void checkParameters() {
        final double[] init = getStartPoint();
        final double[] lB = getLowerBound();
        final double[] uB = getUpperBound();

        if (inputSigma != null) {
            if (inputSigma.length != init.length) {
                throw new DimensionMismatchException(inputSigma.length, init.length);
            }
            for (int i = 0; i < init.length; i++) {
                if (inputSigma[i] > uB[i] - lB[i]) {
                    throw new OutOfRangeException(inputSigma[i], 0, uB[i] - lB[i]);
                }
            }
        }
    }

    
    private void initializeCMA(double[] guess) {
        if (lambda <= 0) {
            throw new NotStrictlyPositiveException(lambda);
        }
        // initialize sigma
        final double[][] sigmaArray = new double[guess.length][1];
        for (int i = 0; i < guess.length; i++) {
            sigmaArray[i][0] = inputSigma[i];
        }
        final RealMatrix insigma = new Array2DRowRealMatrix(sigmaArray, false);
        sigma = max(insigma); // overall standard deviation

        // initialize termination criteria
        stopTolUpX = 1e3 * max(insigma);
        stopTolX = 1e-11 * max(insigma);
        stopTolFun = 1e-12;
        stopTolHistFun = 1e-13;

        // initialize selection strategy parameters
        mu = lambda / 2; // number of parents/points for recombination
        logMu2 = FastMath.log(mu + 0.5);
        weights = log(sequence(1, mu, 1)).scalarMultiply(-1).scalarAdd(logMu2);
        double sumw = 0;
        double sumwq = 0;
        for (int i = 0; i < mu; i++) {
            double w = weights.getEntry(i, 0);
            sumw += w;
            sumwq += w * w;
        }
        weights = weights.scalarMultiply(1 / sumw);
        mueff = sumw * sumw / sumwq; // variance-effectiveness of sum w_i x_i

        // initialize dynamic strategy parameters and constants
        cc = (4 + mueff / dimension) /
                (dimension + 4 + 2 * mueff / dimension);
        cs = (mueff + 2) / (dimension + mueff + 3.);
        damps = (1 + 2 * FastMath.max(0, FastMath.sqrt((mueff - 1) /
                                                       (dimension + 1)) - 1)) *
            FastMath.max(0.3,
                         1 - dimension / (1e-6 + maxIterations)) + cs; // minor increment
        ccov1 = 2 / ((dimension + 1.3) * (dimension + 1.3) + mueff);
        ccovmu = FastMath.min(1 - ccov1, 2 * (mueff - 2 + 1 / mueff) /
                              ((dimension + 2) * (dimension + 2) + mueff));
        ccov1Sep = FastMath.min(1, ccov1 * (dimension + 1.5) / 3);
        ccovmuSep = FastMath.min(1 - ccov1, ccovmu * (dimension + 1.5) / 3);
        chiN = FastMath.sqrt(dimension) *
                (1 - 1 / ((double) 4 * dimension) + 1 / ((double) 21 * dimension * dimension));
        // intialize CMA internal values - updated each generation
        xmean = MatrixUtils.createColumnRealMatrix(guess); // objective variables
        diagD = insigma.scalarMultiply(1 / sigma);
        diagC = square(diagD);
        pc = zeros(dimension, 1); // evolution paths for C and sigma
        ps = zeros(dimension, 1); // B defines the coordinate system
        normps = ps.getFrobeniusNorm();

        B = eye(dimension, dimension);
        D = ones(dimension, 1); // diagonal D defines the scaling
        BD = times(B, repmat(diagD.transpose(), dimension, 1));
        C = B.multiply(diag(square(D)).multiply(B.transpose())); // covariance
        historySize = 10 + (int) (3 * 10 * dimension / (double) lambda);
        fitnessHistory = new double[historySize]; // history of fitness values
        for (int i = 0; i < historySize; i++) {
            fitnessHistory[i] = Double.MAX_VALUE;
        }
    }

    
    private boolean updateEvolutionPaths(RealMatrix zmean, RealMatrix xold) {
        ps = ps.scalarMultiply(1 - cs).add(
                B.multiply(zmean).scalarMultiply(
                        FastMath.sqrt(cs * (2 - cs) * mueff)));
        normps = ps.getFrobeniusNorm();
        final boolean hsig = normps /
            FastMath.sqrt(1 - FastMath.pow(1 - cs, 2 * iterations)) /
            chiN < 1.4 + 2 / ((double) dimension + 1);
        pc = pc.scalarMultiply(1 - cc);
        if (hsig) {
            pc = pc.add(xmean.subtract(xold).scalarMultiply(FastMath.sqrt(cc * (2 - cc) * mueff) / sigma));
        }
        return hsig;
    }

    
    private void updateCovarianceDiagonalOnly(boolean hsig,
                                              final RealMatrix bestArz) {
        // minor correction if hsig==false
        double oldFac = hsig ? 0 : ccov1Sep * cc * (2 - cc);
        oldFac += 1 - ccov1Sep - ccovmuSep;
        diagC = diagC.scalarMultiply(oldFac) // regard old matrix
            .add(square(pc).scalarMultiply(ccov1Sep)) // plus rank one update
            .add((times(diagC, square(bestArz).multiply(weights))) // plus rank mu update
                 .scalarMultiply(ccovmuSep));
        diagD = sqrt(diagC); // replaces eig(C)
        if (diagonalOnly > 1 &&
            iterations > diagonalOnly) {
            // full covariance matrix from now on
            diagonalOnly = 0;
            B = eye(dimension, dimension);
            BD = diag(diagD);
            C = diag(diagC);
        }
    }

    
    private void updateCovariance(boolean hsig, final RealMatrix bestArx,
                                  final RealMatrix arz, final int[] arindex,
                                  final RealMatrix xold) {
        double negccov = 0;
        if (ccov1 + ccovmu > 0) {
            final RealMatrix arpos = bestArx.subtract(repmat(xold, 1, mu))
                .scalarMultiply(1 / sigma); // mu difference vectors
            final RealMatrix roneu = pc.multiply(pc.transpose())
                .scalarMultiply(ccov1); // rank one update
            // minor correction if hsig==false
            double oldFac = hsig ? 0 : ccov1 * cc * (2 - cc);
            oldFac += 1 - ccov1 - ccovmu;
            if (isActiveCMA) {
                // Adapt covariance matrix C active CMA
                negccov = (1 - ccovmu) * 0.25 * mueff /
                    (FastMath.pow(dimension + 2, 1.5) + 2 * mueff);
                // keep at least 0.66 in all directions, small popsize are most
                // critical
                final double negminresidualvariance = 0.66;
                // where to make up for the variance loss
                final double negalphaold = 0.5;
                // prepare vectors, compute negative updating matrix Cneg
                final int[] arReverseIndex = reverse(arindex);
                RealMatrix arzneg = selectColumns(arz, MathArrays.copyOf(arReverseIndex, mu));
                RealMatrix arnorms = sqrt(sumRows(square(arzneg)));
                final int[] idxnorms = sortedIndices(arnorms.getRow(0));
                final RealMatrix arnormsSorted = selectColumns(arnorms, idxnorms);
                final int[] idxReverse = reverse(idxnorms);
                final RealMatrix arnormsReverse = selectColumns(arnorms, idxReverse);
                arnorms = divide(arnormsReverse, arnormsSorted);
                final int[] idxInv = inverse(idxnorms);
                final RealMatrix arnormsInv = selectColumns(arnorms, idxInv);
                // check and set learning rate negccov
                final double negcovMax = (1 - negminresidualvariance) /
                    square(arnormsInv).multiply(weights).getEntry(0, 0);
                if (negccov > negcovMax) {
                    negccov = negcovMax;
                }
                arzneg = times(arzneg, repmat(arnormsInv, dimension, 1));
                final RealMatrix artmp = BD.multiply(arzneg);
                final RealMatrix Cneg = artmp.multiply(diag(weights)).multiply(artmp.transpose());
                oldFac += negalphaold * negccov;
                C = C.scalarMultiply(oldFac)
                    .add(roneu) // regard old matrix
                    .add(arpos.scalarMultiply( // plus rank one update
                                              ccovmu + (1 - negalphaold) * negccov) // plus rank mu update
                         .multiply(times(repmat(weights, 1, dimension),
                                         arpos.transpose())))
                    .subtract(Cneg.scalarMultiply(negccov));
            } else {
                // Adapt covariance matrix C - nonactive
                C = C.scalarMultiply(oldFac) // regard old matrix
                    .add(roneu) // plus rank one update
                    .add(arpos.scalarMultiply(ccovmu) // plus rank mu update
                         .multiply(times(repmat(weights, 1, dimension),
                                         arpos.transpose())));
            }
        }
        updateBD(negccov);
    }

    
    private void updateBD(double negccov) {
        if (ccov1 + ccovmu + negccov > 0 &&
            (iterations % 1. / (ccov1 + ccovmu + negccov) / dimension / 10.) < 1) {
            // to achieve O(N^2)
            C = triu(C, 0).add(triu(C, 1).transpose());
            // enforce symmetry to prevent complex numbers
            final EigenDecomposition eig = new EigenDecomposition(C);
            B = eig.getV(); // eigen decomposition, B==normalized eigenvectors
            D = eig.getD();
            diagD = diag(D);
            if (min(diagD) <= 0) {
                for (int i = 0; i < dimension; i++) {
                    if (diagD.getEntry(i, 0) < 0) {
                        diagD.setEntry(i, 0, 0);
                    }
                }
                final double tfac = max(diagD) / 1e14;
                C = C.add(eye(dimension, dimension).scalarMultiply(tfac));
                diagD = diagD.add(ones(dimension, 1).scalarMultiply(tfac));
            }
            if (max(diagD) > 1e14 * min(diagD)) {
                final double tfac = max(diagD) / 1e14 - min(diagD);
                C = C.add(eye(dimension, dimension).scalarMultiply(tfac));
                diagD = diagD.add(ones(dimension, 1).scalarMultiply(tfac));
            }
            diagC = diag(C);
            diagD = sqrt(diagD); // D contains standard deviations now
            BD = times(B, repmat(diagD.transpose(), dimension, 1)); // O(n^2)
        }
    }

    
    private static void push(double[] vals, double val) {
        for (int i = vals.length-1; i > 0; i--) {
            vals[i] = vals[i-1];
        }
        vals[0] = val;
    }

    
    private int[] sortedIndices(final double[] doubles) {
        final DoubleIndex[] dis = new DoubleIndex[doubles.length];
        for (int i = 0; i < doubles.length; i++) {
            dis[i] = new DoubleIndex(doubles[i], i);
        }
        Arrays.sort(dis);
        final int[] indices = new int[doubles.length];
        for (int i = 0; i < doubles.length; i++) {
            indices[i] = dis[i].index;
        }
        return indices;
    }
   
    private double valueRange(final ValuePenaltyPair[] vpPairs) {
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.MAX_VALUE;
        for (ValuePenaltyPair vpPair:vpPairs) {
            if (vpPair.value > max) {
                max = vpPair.value;
            }
            if (vpPair.value < min) {
                min = vpPair.value;
            }
        }
        return max-min;
    }

    
    private static class DoubleIndex implements Comparable<DoubleIndex> {
        
        private final double value;
        
        private final int index;

        
        DoubleIndex(double value, int index) {
            this.value = value;
            this.index = index;
        }

        
        public int compareTo(DoubleIndex o) {
            return Double.compare(value, o.value);
        }

        
        @Override
        public boolean equals(Object other) {

            if (this == other) {
                return true;
            }

            if (other instanceof DoubleIndex) {
                return Double.compare(value, ((DoubleIndex) other).value) == 0;
            }

            return false;
        }

        
        @Override
        public int hashCode() {
            long bits = Double.doubleToLongBits(value);
            return (int) ((1438542 ^ (bits >>> 32) ^ bits) & 0xffffffff);
        }
    }
    
    private static class ValuePenaltyPair {
        
        private double value;
        
        private double penalty;

        
        ValuePenaltyPair(final double value, final double penalty) {
            this.value   = value;
            this.penalty = penalty;
        }
    }


    
    private class FitnessFunction {
        
        private final boolean isRepairMode;

        
        FitnessFunction() {
            isRepairMode = true;
        }

        
        public ValuePenaltyPair value(final double[] point) {
            double value;
            double penalty=0.0;
            if (isRepairMode) {
                double[] repaired = repair(point);
                value = CMAESOptimizer.this.computeObjectiveValue(repaired);
                penalty =  penalty(point, repaired);
            } else {
                value = CMAESOptimizer.this.computeObjectiveValue(point);
            }
            value = isMinimize ? value : -value;
            penalty = isMinimize ? penalty : -penalty;
            return new ValuePenaltyPair(value,penalty);
        }

        
        public boolean isFeasible(final double[] x) {
            final double[] lB = CMAESOptimizer.this.getLowerBound();
            final double[] uB = CMAESOptimizer.this.getUpperBound();

            for (int i = 0; i < x.length; i++) {
                if (x[i] < lB[i]) {
                    return false;
                }
                if (x[i] > uB[i]) {
                    return false;
                }
            }
            return true;
        }

        
        private double[] repair(final double[] x) {
            final double[] lB = CMAESOptimizer.this.getLowerBound();
            final double[] uB = CMAESOptimizer.this.getUpperBound();

            final double[] repaired = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                if (x[i] < lB[i]) {
                    repaired[i] = lB[i];
                } else if (x[i] > uB[i]) {
                    repaired[i] = uB[i];
                } else {
                    repaired[i] = x[i];
                }
            }
            return repaired;
        }

        
        private double penalty(final double[] x, final double[] repaired) {
            double penalty = 0;
            for (int i = 0; i < x.length; i++) {
                double diff = FastMath.abs(x[i] - repaired[i]);
                penalty += diff;
            }
            return isMinimize ? penalty : -penalty;
        }
    }

    // -----Matrix utility functions similar to the Matlab build in functions------

    
    private static RealMatrix log(final RealMatrix m) {
        final double[][] d = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                d[r][c] = FastMath.log(m.getEntry(r, c));
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix sqrt(final RealMatrix m) {
        final double[][] d = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                d[r][c] = FastMath.sqrt(m.getEntry(r, c));
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix square(final RealMatrix m) {
        final double[][] d = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                double e = m.getEntry(r, c);
                d[r][c] = e * e;
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix times(final RealMatrix m, final RealMatrix n) {
        final double[][] d = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                d[r][c] = m.getEntry(r, c) * n.getEntry(r, c);
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix divide(final RealMatrix m, final RealMatrix n) {
        final double[][] d = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                d[r][c] = m.getEntry(r, c) / n.getEntry(r, c);
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix selectColumns(final RealMatrix m, final int[] cols) {
        final double[][] d = new double[m.getRowDimension()][cols.length];
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < cols.length; c++) {
                d[r][c] = m.getEntry(r, cols[c]);
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix triu(final RealMatrix m, int k) {
        final double[][] d = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                d[r][c] = r <= c - k ? m.getEntry(r, c) : 0;
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix sumRows(final RealMatrix m) {
        final double[][] d = new double[1][m.getColumnDimension()];
        for (int c = 0; c < m.getColumnDimension(); c++) {
            double sum = 0;
            for (int r = 0; r < m.getRowDimension(); r++) {
                sum += m.getEntry(r, c);
            }
            d[0][c] = sum;
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix diag(final RealMatrix m) {
        if (m.getColumnDimension() == 1) {
            final double[][] d = new double[m.getRowDimension()][m.getRowDimension()];
            for (int i = 0; i < m.getRowDimension(); i++) {
                d[i][i] = m.getEntry(i, 0);
            }
            return new Array2DRowRealMatrix(d, false);
        } else {
            final double[][] d = new double[m.getRowDimension()][1];
            for (int i = 0; i < m.getColumnDimension(); i++) {
                d[i][0] = m.getEntry(i, i);
            }
            return new Array2DRowRealMatrix(d, false);
        }
    }

    
    private static void copyColumn(final RealMatrix m1, int col1,
                                   RealMatrix m2, int col2) {
        for (int i = 0; i < m1.getRowDimension(); i++) {
            m2.setEntry(i, col2, m1.getEntry(i, col1));
        }
    }

    
    private static RealMatrix ones(int n, int m) {
        final double[][] d = new double[n][m];
        for (int r = 0; r < n; r++) {
            Arrays.fill(d[r], 1);
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix eye(int n, int m) {
        final double[][] d = new double[n][m];
        for (int r = 0; r < n; r++) {
            if (r < m) {
                d[r][r] = 1;
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix zeros(int n, int m) {
        return new Array2DRowRealMatrix(n, m);
    }

    
    private static RealMatrix repmat(final RealMatrix mat, int n, int m) {
        final int rd = mat.getRowDimension();
        final int cd = mat.getColumnDimension();
        final double[][] d = new double[n * rd][m * cd];
        for (int r = 0; r < n * rd; r++) {
            for (int c = 0; c < m * cd; c++) {
                d[r][c] = mat.getEntry(r % rd, c % cd);
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static RealMatrix sequence(double start, double end, double step) {
        final int size = (int) ((end - start) / step + 1);
        final double[][] d = new double[size][1];
        double value = start;
        for (int r = 0; r < size; r++) {
            d[r][0] = value;
            value += step;
        }
        return new Array2DRowRealMatrix(d, false);
    }

    
    private static double max(final RealMatrix m) {
        double max = -Double.MAX_VALUE;
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                double e = m.getEntry(r, c);
                if (max < e) {
                    max = e;
                }
            }
        }
        return max;
    }

    
    private static double min(final RealMatrix m) {
        double min = Double.MAX_VALUE;
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                double e = m.getEntry(r, c);
                if (min > e) {
                    min = e;
                }
            }
        }
        return min;
    }

    
    private static double max(final double[] m) {
        double max = -Double.MAX_VALUE;
        for (int r = 0; r < m.length; r++) {
            if (max < m[r]) {
                max = m[r];
            }
        }
        return max;
    }

    
    private static double min(final double[] m) {
        double min = Double.MAX_VALUE;
        for (int r = 0; r < m.length; r++) {
            if (min > m[r]) {
                min = m[r];
            }
        }
        return min;
    }

    
    private static int[] inverse(final int[] indices) {
        final int[] inverse = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            inverse[indices[i]] = i;
        }
        return inverse;
    }

    
    private static int[] reverse(final int[] indices) {
        final int[] reverse = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            reverse[i] = indices[indices.length - i - 1];
        }
        return reverse;
    }

    
    private double[] randn(int size) {
        final double[] randn = new double[size];
        for (int i = 0; i < size; i++) {
            randn[i] = random.nextGaussian();
        }
        return randn;
    }

    
    private RealMatrix randn1(int size, int popSize) {
        final double[][] d = new double[size][popSize];
        for (int r = 0; r < size; r++) {
            for (int c = 0; c < popSize; c++) {
                d[r][c] = random.nextGaussian();
            }
        }
        return new Array2DRowRealMatrix(d, false);
    }
}
