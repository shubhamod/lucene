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

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.fraction.BigFraction;
import org.apache.lucene.util.hnsw.math.fraction.Fraction;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class MatrixUtils {

    
    public static final RealMatrixFormat DEFAULT_FORMAT = RealMatrixFormat.getInstance();

    
    public static final RealMatrixFormat OCTAVE_FORMAT = new RealMatrixFormat("[", "]", "", "", "; ", ", ");

    
    private MatrixUtils() {
        super();
    }

    
    public static RealMatrix createRealMatrix(final int rows, final int columns) {
        return (rows * columns <= 4096) ?
                new Array2DRowRealMatrix(rows, columns) : new BlockRealMatrix(rows, columns);
    }

    
    public static <T extends FieldElement<T>> FieldMatrix<T> createFieldMatrix(final Field<T> field,
                                                                               final int rows,
                                                                               final int columns) {
        return (rows * columns <= 4096) ?
                new Array2DRowFieldMatrix<T>(field, rows, columns) : new BlockFieldMatrix<T>(field, rows, columns);
    }

    
    public static RealMatrix createRealMatrix(double[][] data)
        throws NullArgumentException, DimensionMismatchException,
        NoDataException {
        if (data == null ||
            data[0] == null) {
            throw new NullArgumentException();
        }
        return (data.length * data[0].length <= 4096) ?
                new Array2DRowRealMatrix(data) : new BlockRealMatrix(data);
    }

    
    public static <T extends FieldElement<T>> FieldMatrix<T> createFieldMatrix(T[][] data)
        throws DimensionMismatchException, NoDataException, NullArgumentException {
        if (data == null ||
            data[0] == null) {
            throw new NullArgumentException();
        }
        return (data.length * data[0].length <= 4096) ?
                new Array2DRowFieldMatrix<T>(data) : new BlockFieldMatrix<T>(data);
    }

    
    public static RealMatrix createRealIdentityMatrix(int dimension) {
        final RealMatrix m = createRealMatrix(dimension, dimension);
        for (int i = 0; i < dimension; ++i) {
            m.setEntry(i, i, 1.0);
        }
        return m;
    }

    
    public static <T extends FieldElement<T>> FieldMatrix<T>
        createFieldIdentityMatrix(final Field<T> field, final int dimension) {
        final T zero = field.getZero();
        final T one  = field.getOne();
        final T[][] d = MathArrays.buildArray(field, dimension, dimension);
        for (int row = 0; row < dimension; row++) {
            final T[] dRow = d[row];
            Arrays.fill(dRow, zero);
            dRow[row] = one;
        }
        return new Array2DRowFieldMatrix<T>(field, d, false);
    }

    
    public static RealMatrix createRealDiagonalMatrix(final double[] diagonal) {
        final RealMatrix m = createRealMatrix(diagonal.length, diagonal.length);
        for (int i = 0; i < diagonal.length; ++i) {
            m.setEntry(i, i, diagonal[i]);
        }
        return m;
    }

    
    public static <T extends FieldElement<T>> FieldMatrix<T>
        createFieldDiagonalMatrix(final T[] diagonal) {
        final FieldMatrix<T> m =
            createFieldMatrix(diagonal[0].getField(), diagonal.length, diagonal.length);
        for (int i = 0; i < diagonal.length; ++i) {
            m.setEntry(i, i, diagonal[i]);
        }
        return m;
    }

    
    public static RealVector createRealVector(double[] data)
        throws NoDataException, NullArgumentException {
        if (data == null) {
            throw new NullArgumentException();
        }
        return new ArrayRealVector(data, true);
    }

    
    public static <T extends FieldElement<T>> FieldVector<T> createFieldVector(final T[] data)
        throws NoDataException, NullArgumentException, ZeroException {
        if (data == null) {
            throw new NullArgumentException();
        }
        if (data.length == 0) {
            throw new ZeroException(LocalizedFormats.VECTOR_MUST_HAVE_AT_LEAST_ONE_ELEMENT);
        }
        return new ArrayFieldVector<T>(data[0].getField(), data, true);
    }

    
    public static RealMatrix createRowRealMatrix(double[] rowData)
        throws NoDataException, NullArgumentException {
        if (rowData == null) {
            throw new NullArgumentException();
        }
        final int nCols = rowData.length;
        final RealMatrix m = createRealMatrix(1, nCols);
        for (int i = 0; i < nCols; ++i) {
            m.setEntry(0, i, rowData[i]);
        }
        return m;
    }

    
    public static <T extends FieldElement<T>> FieldMatrix<T>
        createRowFieldMatrix(final T[] rowData)
        throws NoDataException, NullArgumentException {
        if (rowData == null) {
            throw new NullArgumentException();
        }
        final int nCols = rowData.length;
        if (nCols == 0) {
            throw new NoDataException(LocalizedFormats.AT_LEAST_ONE_COLUMN);
        }
        final FieldMatrix<T> m = createFieldMatrix(rowData[0].getField(), 1, nCols);
        for (int i = 0; i < nCols; ++i) {
            m.setEntry(0, i, rowData[i]);
        }
        return m;
    }

    
    public static RealMatrix createColumnRealMatrix(double[] columnData)
        throws NoDataException, NullArgumentException {
        if (columnData == null) {
            throw new NullArgumentException();
        }
        final int nRows = columnData.length;
        final RealMatrix m = createRealMatrix(nRows, 1);
        for (int i = 0; i < nRows; ++i) {
            m.setEntry(i, 0, columnData[i]);
        }
        return m;
    }

    
    public static <T extends FieldElement<T>> FieldMatrix<T>
        createColumnFieldMatrix(final T[] columnData)
        throws NoDataException, NullArgumentException {
        if (columnData == null) {
            throw new NullArgumentException();
        }
        final int nRows = columnData.length;
        if (nRows == 0) {
            throw new NoDataException(LocalizedFormats.AT_LEAST_ONE_ROW);
        }
        final FieldMatrix<T> m = createFieldMatrix(columnData[0].getField(), nRows, 1);
        for (int i = 0; i < nRows; ++i) {
            m.setEntry(i, 0, columnData[i]);
        }
        return m;
    }

    
    private static boolean isSymmetricInternal(RealMatrix matrix,
                                               double relativeTolerance,
                                               boolean raiseException) {
        final int rows = matrix.getRowDimension();
        if (rows != matrix.getColumnDimension()) {
            if (raiseException) {
                throw new NonSquareMatrixException(rows, matrix.getColumnDimension());
            } else {
                return false;
            }
        }
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < rows; j++) {
                final double mij = matrix.getEntry(i, j);
                final double mji = matrix.getEntry(j, i);
                if (FastMath.abs(mij - mji) >
                    FastMath.max(FastMath.abs(mij), FastMath.abs(mji)) * relativeTolerance) {
                    if (raiseException) {
                        throw new NonSymmetricMatrixException(i, j, relativeTolerance);
                    } else {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    
    public static void checkSymmetric(RealMatrix matrix,
                                      double eps) {
        isSymmetricInternal(matrix, eps, true);
    }

    
    public static boolean isSymmetric(RealMatrix matrix,
                                      double eps) {
        return isSymmetricInternal(matrix, eps, false);
    }

    
    public static void checkMatrixIndex(final AnyMatrix m,
                                        final int row, final int column)
        throws OutOfRangeException {
        checkRowIndex(m, row);
        checkColumnIndex(m, column);
    }

    
    public static void checkRowIndex(final AnyMatrix m, final int row)
        throws OutOfRangeException {
        if (row < 0 ||
            row >= m.getRowDimension()) {
            throw new OutOfRangeException(LocalizedFormats.ROW_INDEX,
                                          row, 0, m.getRowDimension() - 1);
        }
    }

    
    public static void checkColumnIndex(final AnyMatrix m, final int column)
        throws OutOfRangeException {
        if (column < 0 || column >= m.getColumnDimension()) {
            throw new OutOfRangeException(LocalizedFormats.COLUMN_INDEX,
                                           column, 0, m.getColumnDimension() - 1);
        }
    }

    
    public static void checkSubMatrixIndex(final AnyMatrix m,
                                           final int startRow, final int endRow,
                                           final int startColumn, final int endColumn)
        throws NumberIsTooSmallException, OutOfRangeException {
        checkRowIndex(m, startRow);
        checkRowIndex(m, endRow);
        if (endRow < startRow) {
            throw new NumberIsTooSmallException(LocalizedFormats.INITIAL_ROW_AFTER_FINAL_ROW,
                                                endRow, startRow, false);
        }

        checkColumnIndex(m, startColumn);
        checkColumnIndex(m, endColumn);
        if (endColumn < startColumn) {
            throw new NumberIsTooSmallException(LocalizedFormats.INITIAL_COLUMN_AFTER_FINAL_COLUMN,
                                                endColumn, startColumn, false);
        }


    }

    
    public static void checkSubMatrixIndex(final AnyMatrix m,
                                           final int[] selectedRows,
                                           final int[] selectedColumns)
        throws NoDataException, NullArgumentException, OutOfRangeException {
        if (selectedRows == null) {
            throw new NullArgumentException();
        }
        if (selectedColumns == null) {
            throw new NullArgumentException();
        }
        if (selectedRows.length == 0) {
            throw new NoDataException(LocalizedFormats.EMPTY_SELECTED_ROW_INDEX_ARRAY);
        }
        if (selectedColumns.length == 0) {
            throw new NoDataException(LocalizedFormats.EMPTY_SELECTED_COLUMN_INDEX_ARRAY);
        }

        for (final int row : selectedRows) {
            checkRowIndex(m, row);
        }
        for (final int column : selectedColumns) {
            checkColumnIndex(m, column);
        }
    }

    
    public static void checkAdditionCompatible(final AnyMatrix left, final AnyMatrix right)
        throws MatrixDimensionMismatchException {
        if ((left.getRowDimension()    != right.getRowDimension()) ||
            (left.getColumnDimension() != right.getColumnDimension())) {
            throw new MatrixDimensionMismatchException(left.getRowDimension(), left.getColumnDimension(),
                                                       right.getRowDimension(), right.getColumnDimension());
        }
    }

    
    public static void checkSubtractionCompatible(final AnyMatrix left, final AnyMatrix right)
        throws MatrixDimensionMismatchException {
        if ((left.getRowDimension()    != right.getRowDimension()) ||
            (left.getColumnDimension() != right.getColumnDimension())) {
            throw new MatrixDimensionMismatchException(left.getRowDimension(), left.getColumnDimension(),
                                                       right.getRowDimension(), right.getColumnDimension());
        }
    }

    
    public static void checkMultiplicationCompatible(final AnyMatrix left, final AnyMatrix right)
        throws DimensionMismatchException {

        if (left.getColumnDimension() != right.getRowDimension()) {
            throw new DimensionMismatchException(left.getColumnDimension(),
                                                 right.getRowDimension());
        }
    }

    
    public static Array2DRowRealMatrix fractionMatrixToRealMatrix(final FieldMatrix<Fraction> m) {
        final FractionMatrixConverter converter = new FractionMatrixConverter();
        m.walkInOptimizedOrder(converter);
        return converter.getConvertedMatrix();
    }

    
    private static class FractionMatrixConverter extends DefaultFieldMatrixPreservingVisitor<Fraction> {
        
        private double[][] data;
        
        FractionMatrixConverter() {
            super(Fraction.ZERO);
        }

        
        @Override
        public void start(int rows, int columns,
                          int startRow, int endRow, int startColumn, int endColumn) {
            data = new double[rows][columns];
        }

        
        @Override
        public void visit(int row, int column, Fraction value) {
            data[row][column] = value.doubleValue();
        }

        
        Array2DRowRealMatrix getConvertedMatrix() {
            return new Array2DRowRealMatrix(data, false);
        }

    }

    
    public static Array2DRowRealMatrix bigFractionMatrixToRealMatrix(final FieldMatrix<BigFraction> m) {
        final BigFractionMatrixConverter converter = new BigFractionMatrixConverter();
        m.walkInOptimizedOrder(converter);
        return converter.getConvertedMatrix();
    }

    
    private static class BigFractionMatrixConverter extends DefaultFieldMatrixPreservingVisitor<BigFraction> {
        
        private double[][] data;
        
        BigFractionMatrixConverter() {
            super(BigFraction.ZERO);
        }

        
        @Override
        public void start(int rows, int columns,
                          int startRow, int endRow, int startColumn, int endColumn) {
            data = new double[rows][columns];
        }

        
        @Override
        public void visit(int row, int column, BigFraction value) {
            data[row][column] = value.doubleValue();
        }

        
        Array2DRowRealMatrix getConvertedMatrix() {
            return new Array2DRowRealMatrix(data, false);
        }
    }

    
    public static void serializeRealVector(final RealVector vector,
                                           final ObjectOutputStream oos)
        throws IOException {
        final int n = vector.getDimension();
        oos.writeInt(n);
        for (int i = 0; i < n; ++i) {
            oos.writeDouble(vector.getEntry(i));
        }
    }

    
    public static void deserializeRealVector(final Object instance,
                                             final String fieldName,
                                             final ObjectInputStream ois)
      throws ClassNotFoundException, IOException {
        try {

            // read the vector data
            final int n = ois.readInt();
            final double[] data = new double[n];
            for (int i = 0; i < n; ++i) {
                data[i] = ois.readDouble();
            }

            // create the instance
            final RealVector vector = new ArrayRealVector(data, false);

            // set up the field
            final java.lang.reflect.Field f =
                instance.getClass().getDeclaredField(fieldName);
            f.setAccessible(true);
            f.set(instance, vector);

        } catch (NoSuchFieldException nsfe) {
            IOException ioe = new IOException();
            ioe.initCause(nsfe);
            throw ioe;
        } catch (IllegalAccessException iae) {
            IOException ioe = new IOException();
            ioe.initCause(iae);
            throw ioe;
        }

    }

    
    public static void serializeRealMatrix(final RealMatrix matrix,
                                           final ObjectOutputStream oos)
        throws IOException {
        final int n = matrix.getRowDimension();
        final int m = matrix.getColumnDimension();
        oos.writeInt(n);
        oos.writeInt(m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                oos.writeDouble(matrix.getEntry(i, j));
            }
        }
    }

    
    public static void deserializeRealMatrix(final Object instance,
                                             final String fieldName,
                                             final ObjectInputStream ois)
      throws ClassNotFoundException, IOException {
        try {

            // read the matrix data
            final int n = ois.readInt();
            final int m = ois.readInt();
            final double[][] data = new double[n][m];
            for (int i = 0; i < n; ++i) {
                final double[] dataI = data[i];
                for (int j = 0; j < m; ++j) {
                    dataI[j] = ois.readDouble();
                }
            }

            // create the instance
            final RealMatrix matrix = new Array2DRowRealMatrix(data, false);

            // set up the field
            final java.lang.reflect.Field f =
                instance.getClass().getDeclaredField(fieldName);
            f.setAccessible(true);
            f.set(instance, matrix);

        } catch (NoSuchFieldException nsfe) {
            IOException ioe = new IOException();
            ioe.initCause(nsfe);
            throw ioe;
        } catch (IllegalAccessException iae) {
            IOException ioe = new IOException();
            ioe.initCause(iae);
            throw ioe;
        }
    }

    
    public static void solveLowerTriangularSystem(RealMatrix rm, RealVector b)
        throws DimensionMismatchException, MathArithmeticException,
        NonSquareMatrixException {
        if ((rm == null) || (b == null) || ( rm.getRowDimension() != b.getDimension())) {
            throw new DimensionMismatchException(
                    (rm == null) ? 0 : rm.getRowDimension(),
                    (b == null) ? 0 : b.getDimension());
        }
        if( rm.getColumnDimension() != rm.getRowDimension() ){
            throw new NonSquareMatrixException(rm.getRowDimension(),
                                               rm.getColumnDimension());
        }
        int rows = rm.getRowDimension();
        for( int i = 0 ; i < rows ; i++ ){
            double diag = rm.getEntry(i, i);
            if( FastMath.abs(diag) < Precision.SAFE_MIN ){
                throw new MathArithmeticException(LocalizedFormats.ZERO_DENOMINATOR);
            }
            double bi = b.getEntry(i)/diag;
            b.setEntry(i,  bi );
            for( int j = i+1; j< rows; j++ ){
                b.setEntry(j, b.getEntry(j)-bi*rm.getEntry(j,i)  );
            }
        }
    }

    
    public static void solveUpperTriangularSystem(RealMatrix rm, RealVector b)
        throws DimensionMismatchException, MathArithmeticException,
        NonSquareMatrixException {
        if ((rm == null) || (b == null) || ( rm.getRowDimension() != b.getDimension())) {
            throw new DimensionMismatchException(
                    (rm == null) ? 0 : rm.getRowDimension(),
                    (b == null) ? 0 : b.getDimension());
        }
        if( rm.getColumnDimension() != rm.getRowDimension() ){
            throw new NonSquareMatrixException(rm.getRowDimension(),
                                               rm.getColumnDimension());
        }
        int rows = rm.getRowDimension();
        for( int i = rows-1 ; i >-1 ; i-- ){
            double diag = rm.getEntry(i, i);
            if( FastMath.abs(diag) < Precision.SAFE_MIN ){
                throw new MathArithmeticException(LocalizedFormats.ZERO_DENOMINATOR);
            }
            double bi = b.getEntry(i)/diag;
            b.setEntry(i,  bi );
            for( int j = i-1; j>-1; j-- ){
                b.setEntry(j, b.getEntry(j)-bi*rm.getEntry(j,i)  );
            }
        }
    }

    
    public static RealMatrix blockInverse(RealMatrix m,
                                          int splitIndex) {
        final int n = m.getRowDimension();
        if (m.getColumnDimension() != n) {
            throw new NonSquareMatrixException(m.getRowDimension(),
                                               m.getColumnDimension());
        }

        final int splitIndex1 = splitIndex + 1;

        final RealMatrix a = m.getSubMatrix(0, splitIndex, 0, splitIndex);
        final RealMatrix b = m.getSubMatrix(0, splitIndex, splitIndex1, n - 1);
        final RealMatrix c = m.getSubMatrix(splitIndex1, n - 1, 0, splitIndex);
        final RealMatrix d = m.getSubMatrix(splitIndex1, n - 1, splitIndex1, n - 1);

        final SingularValueDecomposition aDec = new SingularValueDecomposition(a);
        final DecompositionSolver aSolver = aDec.getSolver();
        if (!aSolver.isNonSingular()) {
            throw new SingularMatrixException();
        }
        final RealMatrix aInv = aSolver.getInverse();

        final SingularValueDecomposition dDec = new SingularValueDecomposition(d);
        final DecompositionSolver dSolver = dDec.getSolver();
        if (!dSolver.isNonSingular()) {
            throw new SingularMatrixException();
        }
        final RealMatrix dInv = dSolver.getInverse();

        final RealMatrix tmp1 = a.subtract(b.multiply(dInv).multiply(c));
        final SingularValueDecomposition tmp1Dec = new SingularValueDecomposition(tmp1);
        final DecompositionSolver tmp1Solver = tmp1Dec.getSolver();
        if (!tmp1Solver.isNonSingular()) {
            throw new SingularMatrixException();
        }
        final RealMatrix result00 = tmp1Solver.getInverse();

        final RealMatrix tmp2 = d.subtract(c.multiply(aInv).multiply(b));
        final SingularValueDecomposition tmp2Dec = new SingularValueDecomposition(tmp2);
        final DecompositionSolver tmp2Solver = tmp2Dec.getSolver();
        if (!tmp2Solver.isNonSingular()) {
            throw new SingularMatrixException();
        }
        final RealMatrix result11 = tmp2Solver.getInverse();

        final RealMatrix result01 = aInv.multiply(b).multiply(result11).scalarMultiply(-1);
        final RealMatrix result10 = dInv.multiply(c).multiply(result00).scalarMultiply(-1);

        final RealMatrix result = new Array2DRowRealMatrix(n, n);
        result.setSubMatrix(result00.getData(), 0, 0);
        result.setSubMatrix(result01.getData(), 0, splitIndex1);
        result.setSubMatrix(result10.getData(), splitIndex1, 0);
        result.setSubMatrix(result11.getData(), splitIndex1, splitIndex1);

        return result;
    }

    
    public static RealMatrix inverse(RealMatrix matrix)
            throws NullArgumentException, SingularMatrixException, NonSquareMatrixException {
        return inverse(matrix, 0);
    }

    
    public static RealMatrix inverse(RealMatrix matrix, double threshold)
            throws NullArgumentException, SingularMatrixException, NonSquareMatrixException {

        MathUtils.checkNotNull(matrix);

        if (!matrix.isSquare()) {
            throw new NonSquareMatrixException(matrix.getRowDimension(),
                                               matrix.getColumnDimension());
        }

        if (matrix instanceof DiagonalMatrix) {
            return ((DiagonalMatrix) matrix).inverse(threshold);
        } else {
            QRDecomposition decomposition = new QRDecomposition(matrix, threshold);
            return decomposition.getSolver().getInverse();
        }
    }
}
