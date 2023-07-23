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

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.util.OpenIntToDoubleHashMap;


public class OpenMapRealMatrix extends AbstractRealMatrix
    implements SparseRealMatrix, Serializable {
    
    private static final long serialVersionUID = -5962461716457143437L;
    
    private final int rows;
    
    private final int columns;
    
    private final OpenIntToDoubleHashMap entries;

    
    public OpenMapRealMatrix(int rowDimension, int columnDimension)
        throws NotStrictlyPositiveException, NumberIsTooLargeException {
        super(rowDimension, columnDimension);
        long lRow = rowDimension;
        long lCol = columnDimension;
        if (lRow * lCol >= Integer.MAX_VALUE) {
            throw new NumberIsTooLargeException(lRow * lCol, Integer.MAX_VALUE, false);
        }
        this.rows = rowDimension;
        this.columns = columnDimension;
        this.entries = new OpenIntToDoubleHashMap(0.0);
    }

    
    public OpenMapRealMatrix(OpenMapRealMatrix matrix) {
        this.rows = matrix.rows;
        this.columns = matrix.columns;
        this.entries = new OpenIntToDoubleHashMap(matrix.entries);
    }

    
    @Override
    public OpenMapRealMatrix copy() {
        return new OpenMapRealMatrix(this);
    }

    
    @Override
    public OpenMapRealMatrix createMatrix(int rowDimension, int columnDimension)
        throws NotStrictlyPositiveException, NumberIsTooLargeException {
        return new OpenMapRealMatrix(rowDimension, columnDimension);
    }

    
    @Override
    public int getColumnDimension() {
        return columns;
    }

    
    public OpenMapRealMatrix add(OpenMapRealMatrix m)
        throws MatrixDimensionMismatchException {

        MatrixUtils.checkAdditionCompatible(this, m);

        final OpenMapRealMatrix out = new OpenMapRealMatrix(this);
        for (OpenIntToDoubleHashMap.Iterator iterator = m.entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final int row = iterator.key() / columns;
            final int col = iterator.key() - row * columns;
            out.setEntry(row, col, getEntry(row, col) + iterator.value());
        }

        return out;

    }

    
    @Override
    public OpenMapRealMatrix subtract(final RealMatrix m)
        throws MatrixDimensionMismatchException {
        try {
            return subtract((OpenMapRealMatrix) m);
        } catch (ClassCastException cce) {
            return (OpenMapRealMatrix) super.subtract(m);
        }
    }

    
    public OpenMapRealMatrix subtract(OpenMapRealMatrix m)
        throws MatrixDimensionMismatchException {
        MatrixUtils.checkAdditionCompatible(this, m);

        final OpenMapRealMatrix out = new OpenMapRealMatrix(this);
        for (OpenIntToDoubleHashMap.Iterator iterator = m.entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final int row = iterator.key() / columns;
            final int col = iterator.key() - row * columns;
            out.setEntry(row, col, getEntry(row, col) - iterator.value());
        }

        return out;
    }

    
    @Override
    public RealMatrix multiply(final RealMatrix m)
        throws DimensionMismatchException, NumberIsTooLargeException {
        try {
            return multiply((OpenMapRealMatrix) m);
        } catch (ClassCastException cce) {

            MatrixUtils.checkMultiplicationCompatible(this, m);

            final int outCols = m.getColumnDimension();
            final BlockRealMatrix out = new BlockRealMatrix(rows, outCols);
            for (OpenIntToDoubleHashMap.Iterator iterator = entries.iterator(); iterator.hasNext();) {
                iterator.advance();
                final double value = iterator.value();
                final int key      = iterator.key();
                final int i        = key / columns;
                final int k        = key % columns;
                for (int j = 0; j < outCols; ++j) {
                    out.addToEntry(i, j, value * m.getEntry(k, j));
                }
            }

            return out;
        }
    }

    
    public OpenMapRealMatrix multiply(OpenMapRealMatrix m)
        throws DimensionMismatchException, NumberIsTooLargeException {
        // Safety check.
        MatrixUtils.checkMultiplicationCompatible(this, m);

        final int outCols = m.getColumnDimension();
        OpenMapRealMatrix out = new OpenMapRealMatrix(rows, outCols);
        for (OpenIntToDoubleHashMap.Iterator iterator = entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final double value = iterator.value();
            final int key      = iterator.key();
            final int i        = key / columns;
            final int k        = key % columns;
            for (int j = 0; j < outCols; ++j) {
                final int rightKey = m.computeKey(k, j);
                if (m.entries.containsKey(rightKey)) {
                    final int outKey = out.computeKey(i, j);
                    final double outValue =
                        out.entries.get(outKey) + value * m.entries.get(rightKey);
                    if (outValue == 0.0) {
                        out.entries.remove(outKey);
                    } else {
                        out.entries.put(outKey, outValue);
                    }
                }
            }
        }

        return out;
    }

    
    @Override
    public double getEntry(int row, int column) throws OutOfRangeException {
        MatrixUtils.checkRowIndex(this, row);
        MatrixUtils.checkColumnIndex(this, column);
        return entries.get(computeKey(row, column));
    }

    
    @Override
    public int getRowDimension() {
        return rows;
    }

    
    @Override
    public void setEntry(int row, int column, double value)
        throws OutOfRangeException {
        MatrixUtils.checkRowIndex(this, row);
        MatrixUtils.checkColumnIndex(this, column);
        if (value == 0.0) {
            entries.remove(computeKey(row, column));
        } else {
            entries.put(computeKey(row, column), value);
        }
    }

    
    @Override
    public void addToEntry(int row, int column, double increment)
        throws OutOfRangeException {
        MatrixUtils.checkRowIndex(this, row);
        MatrixUtils.checkColumnIndex(this, column);
        final int key = computeKey(row, column);
        final double value = entries.get(key) + increment;
        if (value == 0.0) {
            entries.remove(key);
        } else {
            entries.put(key, value);
        }
    }

    
    @Override
    public void multiplyEntry(int row, int column, double factor)
        throws OutOfRangeException {
        MatrixUtils.checkRowIndex(this, row);
        MatrixUtils.checkColumnIndex(this, column);
        final int key = computeKey(row, column);
        final double value = entries.get(key) * factor;
        if (value == 0.0) {
            entries.remove(key);
        } else {
            entries.put(key, value);
        }
    }

    
    private int computeKey(int row, int column) {
        return row * columns + column;
    }


}
