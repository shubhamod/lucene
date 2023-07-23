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
package org.apache.lucene.util.hnsw.math.optim.linear;

import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.TooManyIterationsException;
import org.apache.lucene.util.hnsw.math.optim.OptimizationData;
import org.apache.lucene.util.hnsw.math.optim.PointValuePair;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class SimplexSolver extends LinearOptimizer {
    
    static final int DEFAULT_ULPS = 10;

    
    static final double DEFAULT_CUT_OFF = 1e-10;

    
    private static final double DEFAULT_EPSILON = 1.0e-6;

    
    private final double epsilon;

    
    private final int maxUlps;

    
    private final double cutOff;

    
    private PivotSelectionRule pivotSelection;

    
    private SolutionCallback solutionCallback;

    
    public SimplexSolver() {
        this(DEFAULT_EPSILON, DEFAULT_ULPS, DEFAULT_CUT_OFF);
    }

    
    public SimplexSolver(final double epsilon) {
        this(epsilon, DEFAULT_ULPS, DEFAULT_CUT_OFF);
    }

    
    public SimplexSolver(final double epsilon, final int maxUlps) {
        this(epsilon, maxUlps, DEFAULT_CUT_OFF);
    }

    
    public SimplexSolver(final double epsilon, final int maxUlps, final double cutOff) {
        this.epsilon = epsilon;
        this.maxUlps = maxUlps;
        this.cutOff = cutOff;
        this.pivotSelection = PivotSelectionRule.DANTZIG;
    }

    
    @Override
    public PointValuePair optimize(OptimizationData... optData)
        throws TooManyIterationsException {
        // Set up base class and perform computation.
        return super.optimize(optData);
    }

    
    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        // Allow base class to register its own data.
        super.parseOptimizationData(optData);

        // reset the callback before parsing
        solutionCallback = null;

        for (OptimizationData data : optData) {
            if (data instanceof SolutionCallback) {
                solutionCallback = (SolutionCallback) data;
                continue;
            }
            if (data instanceof PivotSelectionRule) {
                pivotSelection = (PivotSelectionRule) data;
                continue;
            }
        }
    }

    
    private Integer getPivotColumn(SimplexTableau tableau) {
        double minValue = 0;
        Integer minPos = null;
        for (int i = tableau.getNumObjectiveFunctions(); i < tableau.getWidth() - 1; i++) {
            final double entry = tableau.getEntry(0, i);
            // check if the entry is strictly smaller than the current minimum
            // do not use a ulp/epsilon check
            if (entry < minValue) {
                minValue = entry;
                minPos = i;

                // Bland's rule: chose the entering column with the lowest index
                if (pivotSelection == PivotSelectionRule.BLAND && isValidPivotColumn(tableau, i)) {
                    break;
                }
            }
        }
        return minPos;
    }

    
    private boolean isValidPivotColumn(SimplexTableau tableau, int col) {
        for (int i = tableau.getNumObjectiveFunctions(); i < tableau.getHeight(); i++) {
            final double entry = tableau.getEntry(i, col);

            // do the same check as in getPivotRow
            if (Precision.compareTo(entry, 0d, cutOff) > 0) {
                return true;
            }
        }
        return false;
    }

    
    private Integer getPivotRow(SimplexTableau tableau, final int col) {
        // create a list of all the rows that tie for the lowest score in the minimum ratio test
        List<Integer> minRatioPositions = new ArrayList<Integer>();
        double minRatio = Double.MAX_VALUE;
        for (int i = tableau.getNumObjectiveFunctions(); i < tableau.getHeight(); i++) {
            final double rhs = tableau.getEntry(i, tableau.getWidth() - 1);
            final double entry = tableau.getEntry(i, col);

            // only consider pivot elements larger than the cutOff threshold
            // selecting others may lead to degeneracy or numerical instabilities
            if (Precision.compareTo(entry, 0d, cutOff) > 0) {
                final double ratio = FastMath.abs(rhs / entry);
                // check if the entry is strictly equal to the current min ratio
                // do not use a ulp/epsilon check
                final int cmp = Double.compare(ratio, minRatio);
                if (cmp == 0) {
                    minRatioPositions.add(i);
                } else if (cmp < 0) {
                    minRatio = ratio;
                    minRatioPositions.clear();
                    minRatioPositions.add(i);
                }
            }
        }

        if (minRatioPositions.size() == 0) {
            return null;
        } else if (minRatioPositions.size() > 1) {
            // there's a degeneracy as indicated by a tie in the minimum ratio test

            // 1. check if there's an artificial variable that can be forced out of the basis
            if (tableau.getNumArtificialVariables() > 0) {
                for (Integer row : minRatioPositions) {
                    for (int i = 0; i < tableau.getNumArtificialVariables(); i++) {
                        int column = i + tableau.getArtificialVariableOffset();
                        final double entry = tableau.getEntry(row, column);
                        if (Precision.equals(entry, 1d, maxUlps) && row.equals(tableau.getBasicRow(column))) {
                            return row;
                        }
                    }
                }
            }

            // 2. apply Bland's rule to prevent cycling:
            //    take the row for which the corresponding basic variable has the smallest index
            //
            // see http://www.stanford.edu/class/msande310/blandrule.pdf
            // see http://en.wikipedia.org/wiki/Bland%27s_rule (not equivalent to the above paper)

            Integer minRow = null;
            int minIndex = tableau.getWidth();
            for (Integer row : minRatioPositions) {
                final int basicVar = tableau.getBasicVariable(row);
                if (basicVar < minIndex) {
                    minIndex = basicVar;
                    minRow = row;
                }
            }
            return minRow;
        }
        return minRatioPositions.get(0);
    }

    
    protected void doIteration(final SimplexTableau tableau)
        throws TooManyIterationsException,
               UnboundedSolutionException {

        incrementIterationCount();

        Integer pivotCol = getPivotColumn(tableau);
        Integer pivotRow = getPivotRow(tableau, pivotCol);
        if (pivotRow == null) {
            throw new UnboundedSolutionException();
        }

        tableau.performRowOperations(pivotCol, pivotRow);
    }

    
    protected void solvePhase1(final SimplexTableau tableau)
        throws TooManyIterationsException,
               UnboundedSolutionException,
               NoFeasibleSolutionException {

        // make sure we're in Phase 1
        if (tableau.getNumArtificialVariables() == 0) {
            return;
        }

        while (!tableau.isOptimal()) {
            doIteration(tableau);
        }

        // if W is not zero then we have no feasible solution
        if (!Precision.equals(tableau.getEntry(0, tableau.getRhsOffset()), 0d, epsilon)) {
            throw new NoFeasibleSolutionException();
        }
    }

    
    @Override
    public PointValuePair doOptimize()
        throws TooManyIterationsException,
               UnboundedSolutionException,
               NoFeasibleSolutionException {

        // reset the tableau to indicate a non-feasible solution in case
        // we do not pass phase 1 successfully
        if (solutionCallback != null) {
            solutionCallback.setTableau(null);
        }

        final SimplexTableau tableau =
            new SimplexTableau(getFunction(),
                               getConstraints(),
                               getGoalType(),
                               isRestrictedToNonNegative(),
                               epsilon,
                               maxUlps);

        solvePhase1(tableau);
        tableau.dropPhase1Objective();

        // after phase 1, we are sure to have a feasible solution
        if (solutionCallback != null) {
            solutionCallback.setTableau(tableau);
        }

        while (!tableau.isOptimal()) {
            doIteration(tableau);
        }

        // check that the solution respects the nonNegative restriction in case
        // the epsilon/cutOff values are too large for the actual linear problem
        // (e.g. with very small constraint coefficients), the solver might actually
        // find a non-valid solution (with negative coefficients).
        final PointValuePair solution = tableau.getSolution();
        if (isRestrictedToNonNegative()) {
            final double[] coeff = solution.getPoint();
            for (int i = 0; i < coeff.length; i++) {
                if (Precision.compareTo(coeff[i], 0, epsilon) < 0) {
                    throw new NoFeasibleSolutionException();
                }
            }
        }
        return solution;
    }
}
