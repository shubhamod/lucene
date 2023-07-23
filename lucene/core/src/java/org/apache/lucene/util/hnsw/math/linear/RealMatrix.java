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

package org.apache.lucene.util.hnsw.math.linear;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;


public interface RealMatrix extends AnyMatrix {

    
    RealMatrix createMatrix(int rowDimension, int columnDimension)
        throws NotStrictlyPositiveException;

    
    RealMatrix copy();

    
    RealMatrix add(RealMatrix m)
        throws MatrixDimensionMismatchException;

    
    RealMatrix subtract(RealMatrix m)
        throws MatrixDimensionMismatchException;

    
    RealMatrix scalarAdd(double d);

    
    RealMatrix scalarMultiply(double d);

    
    RealMatrix multiply(RealMatrix m)
        throws DimensionMismatchException;

    
    RealMatrix preMultiply(RealMatrix m)
        throws DimensionMismatchException;

    
    RealMatrix power(final int p)
        throws NotPositiveException, NonSquareMatrixException;

    
    double[][] getData();

    
    double getNorm();

    
    double getFrobeniusNorm();

    
    RealMatrix getSubMatrix(int startRow, int endRow, int startColumn,
                            int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException;

    
    RealMatrix getSubMatrix(int[] selectedRows, int[] selectedColumns)
        throws NullArgumentException, NoDataException, OutOfRangeException;

    
    void copySubMatrix(int startRow, int endRow, int startColumn,
                       int endColumn, double[][] destination)
        throws OutOfRangeException, NumberIsTooSmallException,
        MatrixDimensionMismatchException;

    
    void copySubMatrix(int[] selectedRows, int[] selectedColumns,
                       double[][] destination)
        throws OutOfRangeException, NullArgumentException, NoDataException,
        MatrixDimensionMismatchException;

   
    void setSubMatrix(double[][] subMatrix, int row, int column)
        throws NoDataException, OutOfRangeException,
        DimensionMismatchException, NullArgumentException;

   
   RealMatrix getRowMatrix(int row) throws OutOfRangeException;

    
    void setRowMatrix(int row, RealMatrix matrix)
        throws OutOfRangeException, MatrixDimensionMismatchException;

    
    RealMatrix getColumnMatrix(int column)
        throws OutOfRangeException;

    
    void setColumnMatrix(int column, RealMatrix matrix)
        throws OutOfRangeException, MatrixDimensionMismatchException;

    
    RealVector getRowVector(int row)
        throws OutOfRangeException;

    
    void setRowVector(int row, RealVector vector)
        throws OutOfRangeException, MatrixDimensionMismatchException;

    
    RealVector getColumnVector(int column)
        throws OutOfRangeException;

    
    void setColumnVector(int column, RealVector vector)
        throws OutOfRangeException, MatrixDimensionMismatchException;

    
    double[] getRow(int row) throws OutOfRangeException;

    
    void setRow(int row, double[] array)
        throws OutOfRangeException, MatrixDimensionMismatchException;

    
    double[] getColumn(int column) throws OutOfRangeException;

    
    void setColumn(int column, double[] array)
        throws OutOfRangeException, MatrixDimensionMismatchException;

    
    double getEntry(int row, int column) throws OutOfRangeException;

    
    void setEntry(int row, int column, double value) throws OutOfRangeException;

    
    void addToEntry(int row, int column, double increment) throws OutOfRangeException;

    
    void multiplyEntry(int row, int column, double factor) throws OutOfRangeException;

    
    RealMatrix transpose();

    
    double getTrace() throws NonSquareMatrixException;

    
    double[] operate(double[] v) throws DimensionMismatchException;

    
    RealVector operate(RealVector v) throws DimensionMismatchException;

    
    double[] preMultiply(double[] v) throws DimensionMismatchException;

    
    RealVector preMultiply(RealVector v) throws DimensionMismatchException;

    
    double walkInRowOrder(RealMatrixChangingVisitor visitor);

    
    double walkInRowOrder(RealMatrixPreservingVisitor visitor);

    
    double walkInRowOrder(RealMatrixChangingVisitor visitor, int startRow,
        int endRow, int startColumn, int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException;

    
    double walkInRowOrder(RealMatrixPreservingVisitor visitor, int startRow,
        int endRow, int startColumn, int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException;

    
    double walkInColumnOrder(RealMatrixChangingVisitor visitor);

    
    double walkInColumnOrder(RealMatrixPreservingVisitor visitor);

    
    double walkInColumnOrder(RealMatrixChangingVisitor visitor, int startRow,
        int endRow, int startColumn, int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException;

    
    double walkInColumnOrder(RealMatrixPreservingVisitor visitor, int startRow,
        int endRow, int startColumn, int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException;

    
    double walkInOptimizedOrder(RealMatrixChangingVisitor visitor);

    
    double walkInOptimizedOrder(RealMatrixPreservingVisitor visitor);

    
    double walkInOptimizedOrder(RealMatrixChangingVisitor visitor,
        int startRow, int endRow, int startColumn, int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException;

    
    double walkInOptimizedOrder(RealMatrixPreservingVisitor visitor,
        int startRow, int endRow, int startColumn, int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException;
}
