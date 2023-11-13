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

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.GoalType;
import org.apache.lucene.util.hnsw.math.optim.PointValuePair;
import org.apache.lucene.util.hnsw.math.util.Precision;


class SimplexTableau implements Serializable {

    
    private static final String NEGATIVE_VAR_COLUMN_LABEL = "x-";

    
    private static final long serialVersionUID = -1369660067587938365L;

    
    private final LinearObjectiveFunction f;

    
    private final List<LinearConstraint> constraints;

    
    private final boolean restrictToNonNegative;

    
    private final List<String> columnLabels = new ArrayList<String>();

    
    private transient Array2DRowRealMatrix tableau;

    
    private final int numDecisionVariables;

    
    private final int numSlackVariables;

    
    private int numArtificialVariables;

    
    private final double epsilon;

    
    private final int maxUlps;

    
    private int[] basicVariables;

    
    private int[] basicRows;

    
    SimplexTableau(final LinearObjectiveFunction f,
                   final Collection<LinearConstraint> constraints,
                   final GoalType goalType,
                   final boolean restrictToNonNegative,
                   final double epsilon) {
        this(f, constraints, goalType, restrictToNonNegative, epsilon, SimplexSolver.DEFAULT_ULPS);
    }

    
    SimplexTableau(final LinearObjectiveFunction f,
                   final Collection<LinearConstraint> constraints,
                   final GoalType goalType,
                   final boolean restrictToNonNegative,
                   final double epsilon,
                   final int maxUlps) {
        this.f                      = f;
        this.constraints            = normalizeConstraints(constraints);
        this.restrictToNonNegative  = restrictToNonNegative;
        this.epsilon                = epsilon;
        this.maxUlps                = maxUlps;
        this.numDecisionVariables   = f.getCoefficients().getDimension() + (restrictToNonNegative ? 0 : 1);
        this.numSlackVariables      = getConstraintTypeCounts(Relationship.LEQ) +
                                      getConstraintTypeCounts(Relationship.GEQ);
        this.numArtificialVariables = getConstraintTypeCounts(Relationship.EQ) +
                                      getConstraintTypeCounts(Relationship.GEQ);
        this.tableau = createTableau(goalType == GoalType.MAXIMIZE);
        // initialize the basic variables for phase 1:
        //   we know that only slack or artificial variables can be basic
        initializeBasicVariables(getSlackVariableOffset());
        initializeColumnLabels();
    }

    
    protected void initializeColumnLabels() {
      if (getNumObjectiveFunctions() == 2) {
        columnLabels.add("W");
      }
      columnLabels.add("Z");
      for (int i = 0; i < getOriginalNumDecisionVariables(); i++) {
        columnLabels.add("x" + i);
      }
      if (!restrictToNonNegative) {
        columnLabels.add(NEGATIVE_VAR_COLUMN_LABEL);
      }
      for (int i = 0; i < getNumSlackVariables(); i++) {
        columnLabels.add("s" + i);
      }
      for (int i = 0; i < getNumArtificialVariables(); i++) {
        columnLabels.add("a" + i);
      }
      columnLabels.add("RHS");
    }

    
    protected Array2DRowRealMatrix createTableau(final boolean maximize) {

        // create a matrix of the correct size
        int width = numDecisionVariables + numSlackVariables +
        numArtificialVariables + getNumObjectiveFunctions() + 1; // + 1 is for RHS
        int height = constraints.size() + getNumObjectiveFunctions();
        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(height, width);

        // initialize the objective function rows
        if (getNumObjectiveFunctions() == 2) {
            matrix.setEntry(0, 0, -1);
        }

        int zIndex = (getNumObjectiveFunctions() == 1) ? 0 : 1;
        matrix.setEntry(zIndex, zIndex, maximize ? 1 : -1);
        RealVector objectiveCoefficients = maximize ? f.getCoefficients().mapMultiply(-1) : f.getCoefficients();
        copyArray(objectiveCoefficients.toArray(), matrix.getDataRef()[zIndex]);
        matrix.setEntry(zIndex, width - 1, maximize ? f.getConstantTerm() : -1 * f.getConstantTerm());

        if (!restrictToNonNegative) {
            matrix.setEntry(zIndex, getSlackVariableOffset() - 1,
                            getInvertedCoefficientSum(objectiveCoefficients));
        }

        // initialize the constraint rows
        int slackVar = 0;
        int artificialVar = 0;
        for (int i = 0; i < constraints.size(); i++) {
            LinearConstraint constraint = constraints.get(i);
            int row = getNumObjectiveFunctions() + i;

            // decision variable coefficients
            copyArray(constraint.getCoefficients().toArray(), matrix.getDataRef()[row]);

            // x-
            if (!restrictToNonNegative) {
                matrix.setEntry(row, getSlackVariableOffset() - 1,
                                getInvertedCoefficientSum(constraint.getCoefficients()));
            }

            // RHS
            matrix.setEntry(row, width - 1, constraint.getValue());

            // slack variables
            if (constraint.getRelationship() == Relationship.LEQ) {
                matrix.setEntry(row, getSlackVariableOffset() + slackVar++, 1);  // slack
            } else if (constraint.getRelationship() == Relationship.GEQ) {
                matrix.setEntry(row, getSlackVariableOffset() + slackVar++, -1); // excess
            }

            // artificial variables
            if ((constraint.getRelationship() == Relationship.EQ) ||
                (constraint.getRelationship() == Relationship.GEQ)) {
                matrix.setEntry(0, getArtificialVariableOffset() + artificialVar, 1);
                matrix.setEntry(row, getArtificialVariableOffset() + artificialVar++, 1);
                matrix.setRowVector(0, matrix.getRowVector(0).subtract(matrix.getRowVector(row)));
            }
        }

        return matrix;
    }

    
    public List<LinearConstraint> normalizeConstraints(Collection<LinearConstraint> originalConstraints) {
        List<LinearConstraint> normalized = new ArrayList<LinearConstraint>(originalConstraints.size());
        for (LinearConstraint constraint : originalConstraints) {
            normalized.add(normalize(constraint));
        }
        return normalized;
    }

    
    private LinearConstraint normalize(final LinearConstraint constraint) {
        if (constraint.getValue() < 0) {
            return new LinearConstraint(constraint.getCoefficients().mapMultiply(-1),
                                        constraint.getRelationship().oppositeRelationship(),
                                        -1 * constraint.getValue());
        }
        return new LinearConstraint(constraint.getCoefficients(),
                                    constraint.getRelationship(), constraint.getValue());
    }

    
    protected final int getNumObjectiveFunctions() {
        return this.numArtificialVariables > 0 ? 2 : 1;
    }

    
    private int getConstraintTypeCounts(final Relationship relationship) {
        int count = 0;
        for (final LinearConstraint constraint : constraints) {
            if (constraint.getRelationship() == relationship) {
                ++count;
            }
        }
        return count;
    }

    
    protected static double getInvertedCoefficientSum(final RealVector coefficients) {
        double sum = 0;
        for (double coefficient : coefficients.toArray()) {
            sum -= coefficient;
        }
        return sum;
    }

    
    protected Integer getBasicRow(final int col) {
        final int row = basicVariables[col];
        return row == -1 ? null : row;
    }

    
    protected int getBasicVariable(final int row) {
        return basicRows[row];
    }

    
    private void initializeBasicVariables(final int startColumn) {
        basicVariables = new int[getWidth() - 1];
        basicRows = new int[getHeight()];

        Arrays.fill(basicVariables, -1);

        for (int i = startColumn; i < getWidth() - 1; i++) {
            Integer row = findBasicRow(i);
            if (row != null) {
                basicVariables[i] = row;
                basicRows[row] = i;
            }
        }
    }

    
    private Integer findBasicRow(final int col) {
        Integer row = null;
        for (int i = 0; i < getHeight(); i++) {
            final double entry = getEntry(i, col);
            if (Precision.equals(entry, 1d, maxUlps) && (row == null)) {
                row = i;
            } else if (!Precision.equals(entry, 0d, maxUlps)) {
                return null;
            }
        }
        return row;
    }

    
    protected void dropPhase1Objective() {
        if (getNumObjectiveFunctions() == 1) {
            return;
        }

        final Set<Integer> columnsToDrop = new TreeSet<Integer>();
        columnsToDrop.add(0);

        // positive cost non-artificial variables
        for (int i = getNumObjectiveFunctions(); i < getArtificialVariableOffset(); i++) {
            final double entry = getEntry(0, i);
            if (Precision.compareTo(entry, 0d, epsilon) > 0) {
                columnsToDrop.add(i);
            }
        }

        // non-basic artificial variables
        for (int i = 0; i < getNumArtificialVariables(); i++) {
            int col = i + getArtificialVariableOffset();
            if (getBasicRow(col) == null) {
                columnsToDrop.add(col);
            }
        }

        final double[][] matrix = new double[getHeight() - 1][getWidth() - columnsToDrop.size()];
        for (int i = 1; i < getHeight(); i++) {
            int col = 0;
            for (int j = 0; j < getWidth(); j++) {
                if (!columnsToDrop.contains(j)) {
                    matrix[i - 1][col++] = getEntry(i, j);
                }
            }
        }

        // remove the columns in reverse order so the indices are correct
        Integer[] drop = columnsToDrop.toArray(new Integer[columnsToDrop.size()]);
        for (int i = drop.length - 1; i >= 0; i--) {
            columnLabels.remove((int) drop[i]);
        }

        this.tableau = new Array2DRowRealMatrix(matrix);
        this.numArtificialVariables = 0;
        // need to update the basic variable mappings as row/columns have been dropped
        initializeBasicVariables(getNumObjectiveFunctions());
    }

    
    private void copyArray(final double[] src, final double[] dest) {
        System.arraycopy(src, 0, dest, getNumObjectiveFunctions(), src.length);
    }

    
    boolean isOptimal() {
        final double[] objectiveFunctionRow = getRow(0);
        final int end = getRhsOffset();
        for (int i = getNumObjectiveFunctions(); i < end; i++) {
            final double entry = objectiveFunctionRow[i];
            if (Precision.compareTo(entry, 0d, epsilon) < 0) {
                return false;
            }
        }
        return true;
    }

    
    protected PointValuePair getSolution() {
        int negativeVarColumn = columnLabels.indexOf(NEGATIVE_VAR_COLUMN_LABEL);
        Integer negativeVarBasicRow = negativeVarColumn > 0 ? getBasicRow(negativeVarColumn) : null;
        double mostNegative = negativeVarBasicRow == null ? 0 : getEntry(negativeVarBasicRow, getRhsOffset());

        final Set<Integer> usedBasicRows = new HashSet<Integer>();
        final double[] coefficients = new double[getOriginalNumDecisionVariables()];
        for (int i = 0; i < coefficients.length; i++) {
            int colIndex = columnLabels.indexOf("x" + i);
            if (colIndex < 0) {
                coefficients[i] = 0;
                continue;
            }
            Integer basicRow = getBasicRow(colIndex);
            if (basicRow != null && basicRow == 0) {
                // if the basic row is found to be the objective function row
                // set the coefficient to 0 -> this case handles unconstrained
                // variables that are still part of the objective function
                coefficients[i] = 0;
            } else if (usedBasicRows.contains(basicRow)) {
                // if multiple variables can take a given value
                // then we choose the first and set the rest equal to 0
                coefficients[i] = 0 - (restrictToNonNegative ? 0 : mostNegative);
            } else {
                usedBasicRows.add(basicRow);
                coefficients[i] =
                    (basicRow == null ? 0 : getEntry(basicRow, getRhsOffset())) -
                    (restrictToNonNegative ? 0 : mostNegative);
            }
        }
        return new PointValuePair(coefficients, f.value(coefficients));
    }

    
    protected void performRowOperations(int pivotCol, int pivotRow) {
        // set the pivot element to 1
        final double pivotVal = getEntry(pivotRow, pivotCol);
        divideRow(pivotRow, pivotVal);

        // set the rest of the pivot column to 0
        for (int i = 0; i < getHeight(); i++) {
            if (i != pivotRow) {
                final double multiplier = getEntry(i, pivotCol);
                if (multiplier != 0.0) {
                    subtractRow(i, pivotRow, multiplier);
                }
            }
        }

        // update the basic variable mappings
        final int previousBasicVariable = getBasicVariable(pivotRow);
        basicVariables[previousBasicVariable] = -1;
        basicVariables[pivotCol] = pivotRow;
        basicRows[pivotRow] = pivotCol;
    }

    
    protected void divideRow(final int dividendRowIndex, final double divisor) {
        final double[] dividendRow = getRow(dividendRowIndex);
        for (int j = 0; j < getWidth(); j++) {
            dividendRow[j] /= divisor;
        }
    }

    
    protected void subtractRow(final int minuendRowIndex, final int subtrahendRowIndex, final double multiplier) {
        final double[] minuendRow = getRow(minuendRowIndex);
        final double[] subtrahendRow = getRow(subtrahendRowIndex);
        for (int i = 0; i < getWidth(); i++) {
            minuendRow[i] -= subtrahendRow[i] * multiplier;
        }
    }

    
    protected final int getWidth() {
        return tableau.getColumnDimension();
    }

    
    protected final int getHeight() {
        return tableau.getRowDimension();
    }

    
    protected final double getEntry(final int row, final int column) {
        return tableau.getEntry(row, column);
    }

    
    protected final void setEntry(final int row, final int column, final double value) {
        tableau.setEntry(row, column, value);
    }

    
    protected final int getSlackVariableOffset() {
        return getNumObjectiveFunctions() + numDecisionVariables;
    }

    
    protected final int getArtificialVariableOffset() {
        return getNumObjectiveFunctions() + numDecisionVariables + numSlackVariables;
    }

    
    protected final int getRhsOffset() {
        return getWidth() - 1;
    }

    
    protected final int getNumDecisionVariables() {
        return numDecisionVariables;
    }

    
    protected final int getOriginalNumDecisionVariables() {
        return f.getCoefficients().getDimension();
    }

    
    protected final int getNumSlackVariables() {
        return numSlackVariables;
    }

    
    protected final int getNumArtificialVariables() {
        return numArtificialVariables;
    }

    
    protected final double[] getRow(int row) {
        return tableau.getDataRef()[row];
    }

    
    protected final double[][] getData() {
        return tableau.getData();
    }

    
    @Override
    public boolean equals(Object other) {

      if (this == other) {
        return true;
      }

      if (other instanceof SimplexTableau) {
          SimplexTableau rhs = (SimplexTableau) other;
          return (restrictToNonNegative  == rhs.restrictToNonNegative) &&
                 (numDecisionVariables   == rhs.numDecisionVariables) &&
                 (numSlackVariables      == rhs.numSlackVariables) &&
                 (numArtificialVariables == rhs.numArtificialVariables) &&
                 (epsilon                == rhs.epsilon) &&
                 (maxUlps                == rhs.maxUlps) &&
                 f.equals(rhs.f) &&
                 constraints.equals(rhs.constraints) &&
                 tableau.equals(rhs.tableau);
      }
      return false;
    }

    
    @Override
    public int hashCode() {
        return Boolean.valueOf(restrictToNonNegative).hashCode() ^
               numDecisionVariables ^
               numSlackVariables ^
               numArtificialVariables ^
               Double.valueOf(epsilon).hashCode() ^
               maxUlps ^
               f.hashCode() ^
               constraints.hashCode() ^
               tableau.hashCode();
    }

    
    private void writeObject(ObjectOutputStream oos)
        throws IOException {
        oos.defaultWriteObject();
        MatrixUtils.serializeRealMatrix(tableau, oos);
    }

    
    private void readObject(ObjectInputStream ois)
      throws ClassNotFoundException, IOException {
        ois.defaultReadObject();
        MatrixUtils.deserializeRealMatrix(this, "tableau", ois);
    }
}
