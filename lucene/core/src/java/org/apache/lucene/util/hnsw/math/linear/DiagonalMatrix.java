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
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class DiagonalMatrix extends AbstractRealMatrix
    implements Serializable {
    
    private static final long serialVersionUID = 20121229L;
    
    private final double[] data;

    
    public DiagonalMatrix(final int dimension)
        throws NotStrictlyPositiveException {
        super(dimension, dimension);
        data = new double[dimension];
    }

    
    public DiagonalMatrix(final double[] d) {
        this(d, true);
    }

    
    public DiagonalMatrix(final double[] d, final boolean copyArray)
        throws NullArgumentException {
        MathUtils.checkNotNull(d);
        data = copyArray ? d.clone() : d;
    }

    
    @Override
    public RealMatrix createMatrix(final int rowDimension,
                                   final int columnDimension)
        throws NotStrictlyPositiveException,
               DimensionMismatchException {
        if (rowDimension != columnDimension) {
            throw new DimensionMismatchException(rowDimension, columnDimension);
        }

        return new DiagonalMatrix(rowDimension);
    }

    
    @Override
    public RealMatrix copy() {
        return new DiagonalMatrix(data);
    }

    
    public DiagonalMatrix add(final DiagonalMatrix m)
        throws MatrixDimensionMismatchException {
        // Safety check.
        MatrixUtils.checkAdditionCompatible(this, m);

        final int dim = getRowDimension();
        final double[] outData = new double[dim];
        for (int i = 0; i < dim; i++) {
            outData[i] = data[i] + m.data[i];
        }

        return new DiagonalMatrix(outData, false);
    }

    
    public DiagonalMatrix subtract(final DiagonalMatrix m)
        throws MatrixDimensionMismatchException {
        MatrixUtils.checkSubtractionCompatible(this, m);

        final int dim = getRowDimension();
        final double[] outData = new double[dim];
        for (int i = 0; i < dim; i++) {
            outData[i] = data[i] - m.data[i];
        }

        return new DiagonalMatrix(outData, false);
    }

    
    public DiagonalMatrix multiply(final DiagonalMatrix m)
        throws DimensionMismatchException {
        MatrixUtils.checkMultiplicationCompatible(this, m);

        final int dim = getRowDimension();
        final double[] outData = new double[dim];
        for (int i = 0; i < dim; i++) {
            outData[i] = data[i] * m.data[i];
        }

        return new DiagonalMatrix(outData, false);
    }

    
    @Override
    public RealMatrix multiply(final RealMatrix m)
        throws DimensionMismatchException {
        if (m instanceof DiagonalMatrix) {
            return multiply((DiagonalMatrix) m);
        } else {
            MatrixUtils.checkMultiplicationCompatible(this, m);
            final int nRows = m.getRowDimension();
            final int nCols = m.getColumnDimension();
            final double[][] product = new double[nRows][nCols];
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    product[r][c] = data[r] * m.getEntry(r, c);
                }
            }
            return new Array2DRowRealMatrix(product, false);
        }
    }

    
    @Override
    public double[][] getData() {
        final int dim = getRowDimension();
        final double[][] out = new double[dim][dim];

        for (int i = 0; i < dim; i++) {
            out[i][i] = data[i];
        }

        return out;
    }

    
    public double[] getDataRef() {
        return data;
    }

    
    @Override
    public double getEntry(final int row, final int column)
        throws OutOfRangeException {
        MatrixUtils.checkMatrixIndex(this, row, column);
        return row == column ? data[row] : 0;
    }

    
    @Override
    public void setEntry(final int row, final int column, final double value)
        throws OutOfRangeException, NumberIsTooLargeException {
        if (row == column) {
            MatrixUtils.checkRowIndex(this, row);
            data[row] = value;
        } else {
            ensureZero(value);
        }
    }

    
    @Override
    public void addToEntry(final int row,
                           final int column,
                           final double increment)
        throws OutOfRangeException, NumberIsTooLargeException {
        if (row == column) {
            MatrixUtils.checkRowIndex(this, row);
            data[row] += increment;
        } else {
            ensureZero(increment);
        }
    }

    
    @Override
    public void multiplyEntry(final int row,
                              final int column,
                              final double factor)
        throws OutOfRangeException {
        // we don't care about non-diagonal elements for multiplication
        if (row == column) {
            MatrixUtils.checkRowIndex(this, row);
            data[row] *= factor;
        }
    }

    
    @Override
    public int getRowDimension() {
        return data.length;
    }

    
    @Override
    public int getColumnDimension() {
        return data.length;
    }

    
    @Override
    public double[] operate(final double[] v)
        throws DimensionMismatchException {
        return multiply(new DiagonalMatrix(v, false)).getDataRef();
    }

    
    @Override
    public double[] preMultiply(final double[] v)
        throws DimensionMismatchException {
        return operate(v);
    }

    
    @Override
    public RealVector preMultiply(final RealVector v) throws DimensionMismatchException {
        final double[] vectorData;
        if (v instanceof ArrayRealVector) {
            vectorData = ((ArrayRealVector) v).getDataRef();
        } else {
            vectorData = v.toArray();
        }
        return MatrixUtils.createRealVector(preMultiply(vectorData));
    }

    
    private void ensureZero(final double value) throws NumberIsTooLargeException {
        if (!Precision.equals(0.0, value, 1)) {
            throw new NumberIsTooLargeException(FastMath.abs(value), 0, true);
        }
    }

    
    public DiagonalMatrix inverse() throws SingularMatrixException {
        return inverse(0);
    }

    
    public DiagonalMatrix inverse(double threshold) throws SingularMatrixException {
        if (isSingular(threshold)) {
            throw new SingularMatrixException();
        }

        final double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = 1.0 / data[i];
        }
        return new DiagonalMatrix(result, false);
    }

    
    public boolean isSingular(double threshold) {
        for (int i = 0; i < data.length; i++) {
            if (Precision.equals(data[i], 0.0, threshold)) {
                return true;
            }
        }
        return false;
    }
}
