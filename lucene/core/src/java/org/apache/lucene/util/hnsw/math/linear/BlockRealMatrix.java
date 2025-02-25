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
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class BlockRealMatrix extends AbstractRealMatrix implements Serializable {
    
    public static final int BLOCK_SIZE = 52;
    
    private static final long serialVersionUID = 4991895511313664478L;
    
    private final double blocks[][];
    
    private final int rows;
    
    private final int columns;
    
    private final int blockRows;
    
    private final int blockColumns;

    
    public BlockRealMatrix(final int rows, final int columns)
        throws NotStrictlyPositiveException {
        super(rows, columns);
        this.rows = rows;
        this.columns = columns;

        // number of blocks
        blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // allocate storage blocks, taking care of smaller ones at right and bottom
        blocks = createBlocksLayout(rows, columns);
    }

    
    public BlockRealMatrix(final double[][] rawData)
        throws DimensionMismatchException, NotStrictlyPositiveException {
        this(rawData.length, rawData[0].length, toBlocksLayout(rawData), false);
    }

    
    public BlockRealMatrix(final int rows, final int columns,
                           final double[][] blockData, final boolean copyArray)
        throws DimensionMismatchException, NotStrictlyPositiveException {
        super(rows, columns);
        this.rows = rows;
        this.columns = columns;

        // number of blocks
        blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (copyArray) {
            // allocate storage blocks, taking care of smaller ones at right and bottom
            blocks = new double[blockRows * blockColumns][];
        } else {
            // reference existing array
            blocks = blockData;
        }

        int index = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int iHeight = blockHeight(iBlock);
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock, ++index) {
                if (blockData[index].length != iHeight * blockWidth(jBlock)) {
                    throw new DimensionMismatchException(blockData[index].length,
                                                         iHeight * blockWidth(jBlock));
                }
                if (copyArray) {
                    blocks[index] = blockData[index].clone();
                }
            }
        }
    }

    
    public static double[][] toBlocksLayout(final double[][] rawData)
        throws DimensionMismatchException {
        final int rows = rawData.length;
        final int columns = rawData[0].length;
        final int blockRows = (rows    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        final int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // safety checks
        for (int i = 0; i < rawData.length; ++i) {
            final int length = rawData[i].length;
            if (length != columns) {
                throw new DimensionMismatchException(columns, length);
            }
        }

        // convert array
        final double[][] blocks = new double[blockRows * blockColumns][];
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
            final int iHeight = pEnd - pStart;
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
                final int jWidth = qEnd - qStart;

                // allocate new block
                final double[] block = new double[iHeight * jWidth];
                blocks[blockIndex] = block;

                // copy data
                int index = 0;
                for (int p = pStart; p < pEnd; ++p) {
                    System.arraycopy(rawData[p], qStart, block, index, jWidth);
                    index += jWidth;
                }
                ++blockIndex;
            }
        }

        return blocks;
    }

    
    public static double[][] createBlocksLayout(final int rows, final int columns) {
        final int blockRows = (rows    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        final int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;

        final double[][] blocks = new double[blockRows * blockColumns][];
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
            final int iHeight = pEnd - pStart;
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
                final int jWidth = qEnd - qStart;
                blocks[blockIndex] = new double[iHeight * jWidth];
                ++blockIndex;
            }
        }

        return blocks;
    }

    
    @Override
    public BlockRealMatrix createMatrix(final int rowDimension,
                                        final int columnDimension)
        throws NotStrictlyPositiveException {
        return new BlockRealMatrix(rowDimension, columnDimension);
    }

    
    @Override
    public BlockRealMatrix copy() {
        // create an empty matrix
        BlockRealMatrix copied = new BlockRealMatrix(rows, columns);

        // copy the blocks
        for (int i = 0; i < blocks.length; ++i) {
            System.arraycopy(blocks[i], 0, copied.blocks[i], 0, blocks[i].length);
        }

        return copied;
    }

    
    @Override
    public BlockRealMatrix add(final RealMatrix m)
        throws MatrixDimensionMismatchException {
        try {
            return add((BlockRealMatrix) m);
        } catch (ClassCastException cce) {
            // safety check
            MatrixUtils.checkAdditionCompatible(this, m);

            final BlockRealMatrix out = new BlockRealMatrix(rows, columns);

            // perform addition block-wise, to ensure good cache behavior
            int blockIndex = 0;
            for (int iBlock = 0; iBlock < out.blockRows; ++iBlock) {
                for (int jBlock = 0; jBlock < out.blockColumns; ++jBlock) {

                    // perform addition on the current block
                    final double[] outBlock = out.blocks[blockIndex];
                    final double[] tBlock   = blocks[blockIndex];
                    final int pStart = iBlock * BLOCK_SIZE;
                    final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
                    int k = 0;
                    for (int p = pStart; p < pEnd; ++p) {
                        for (int q = qStart; q < qEnd; ++q) {
                            outBlock[k] = tBlock[k] + m.getEntry(p, q);
                            ++k;
                        }
                    }
                    // go to next block
                    ++blockIndex;
                }
            }

            return out;
        }
    }

    
    public BlockRealMatrix add(final BlockRealMatrix m)
        throws MatrixDimensionMismatchException {
        // safety check
        MatrixUtils.checkAdditionCompatible(this, m);

        final BlockRealMatrix out = new BlockRealMatrix(rows, columns);

        // perform addition block-wise, to ensure good cache behavior
        for (int blockIndex = 0; blockIndex < out.blocks.length; ++blockIndex) {
            final double[] outBlock = out.blocks[blockIndex];
            final double[] tBlock = blocks[blockIndex];
            final double[] mBlock = m.blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] + mBlock[k];
            }
        }

        return out;
    }

    
    @Override
    public BlockRealMatrix subtract(final RealMatrix m)
        throws MatrixDimensionMismatchException {
        try {
            return subtract((BlockRealMatrix) m);
        } catch (ClassCastException cce) {
            // safety check
            MatrixUtils.checkSubtractionCompatible(this, m);

            final BlockRealMatrix out = new BlockRealMatrix(rows, columns);

            // perform subtraction block-wise, to ensure good cache behavior
            int blockIndex = 0;
            for (int iBlock = 0; iBlock < out.blockRows; ++iBlock) {
                for (int jBlock = 0; jBlock < out.blockColumns; ++jBlock) {

                    // perform subtraction on the current block
                    final double[] outBlock = out.blocks[blockIndex];
                    final double[] tBlock = blocks[blockIndex];
                    final int pStart = iBlock * BLOCK_SIZE;
                    final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
                    int k = 0;
                    for (int p = pStart; p < pEnd; ++p) {
                        for (int q = qStart; q < qEnd; ++q) {
                            outBlock[k] = tBlock[k] - m.getEntry(p, q);
                            ++k;
                        }
                    }
                    // go to next block
                    ++blockIndex;
                }
            }

            return out;
        }
    }

    
    public BlockRealMatrix subtract(final BlockRealMatrix m)
        throws MatrixDimensionMismatchException {
        // safety check
        MatrixUtils.checkSubtractionCompatible(this, m);

        final BlockRealMatrix out = new BlockRealMatrix(rows, columns);

        // perform subtraction block-wise, to ensure good cache behavior
        for (int blockIndex = 0; blockIndex < out.blocks.length; ++blockIndex) {
            final double[] outBlock = out.blocks[blockIndex];
            final double[] tBlock = blocks[blockIndex];
            final double[] mBlock = m.blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] - mBlock[k];
            }
        }

        return out;
    }

    
    @Override
    public BlockRealMatrix scalarAdd(final double d) {

        final BlockRealMatrix out = new BlockRealMatrix(rows, columns);

        // perform subtraction block-wise, to ensure good cache behavior
        for (int blockIndex = 0; blockIndex < out.blocks.length; ++blockIndex) {
            final double[] outBlock = out.blocks[blockIndex];
            final double[] tBlock = blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] + d;
            }
        }

        return out;
    }

    
    @Override
    public RealMatrix scalarMultiply(final double d) {
        final BlockRealMatrix out = new BlockRealMatrix(rows, columns);

        // perform subtraction block-wise, to ensure good cache behavior
        for (int blockIndex = 0; blockIndex < out.blocks.length; ++blockIndex) {
            final double[] outBlock = out.blocks[blockIndex];
            final double[] tBlock = blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] * d;
            }
        }

        return out;
    }

    
    @Override
    public BlockRealMatrix multiply(final RealMatrix m)
        throws DimensionMismatchException {
        try {
            return multiply((BlockRealMatrix) m);
        } catch (ClassCastException cce) {
            // safety check
            MatrixUtils.checkMultiplicationCompatible(this, m);

            final BlockRealMatrix out = new BlockRealMatrix(rows, m.getColumnDimension());

            // perform multiplication block-wise, to ensure good cache behavior
            int blockIndex = 0;
            for (int iBlock = 0; iBlock < out.blockRows; ++iBlock) {
                final int pStart = iBlock * BLOCK_SIZE;
                final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);

                for (int jBlock = 0; jBlock < out.blockColumns; ++jBlock) {
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = FastMath.min(qStart + BLOCK_SIZE, m.getColumnDimension());

                    // select current block
                    final double[] outBlock = out.blocks[blockIndex];

                    // perform multiplication on current block
                    for (int kBlock = 0; kBlock < blockColumns; ++kBlock) {
                        final int kWidth = blockWidth(kBlock);
                        final double[] tBlock = blocks[iBlock * blockColumns + kBlock];
                        final int rStart = kBlock * BLOCK_SIZE;
                        int k = 0;
                        for (int p = pStart; p < pEnd; ++p) {
                            final int lStart = (p - pStart) * kWidth;
                            final int lEnd = lStart + kWidth;
                            for (int q = qStart; q < qEnd; ++q) {
                                double sum = 0;
                                int r = rStart;
                                for (int l = lStart; l < lEnd; ++l) {
                                    sum += tBlock[l] * m.getEntry(r, q);
                                    ++r;
                                }
                                outBlock[k] += sum;
                                ++k;
                            }
                        }
                    }
                    // go to next block
                    ++blockIndex;
                }
            }

            return out;
        }
    }

    
    public BlockRealMatrix multiply(BlockRealMatrix m)
        throws DimensionMismatchException {
        // safety check
        MatrixUtils.checkMultiplicationCompatible(this, m);

        final BlockRealMatrix out = new BlockRealMatrix(rows, m.columns);

        // perform multiplication block-wise, to ensure good cache behavior
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < out.blockRows; ++iBlock) {

            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);

            for (int jBlock = 0; jBlock < out.blockColumns; ++jBlock) {
                final int jWidth = out.blockWidth(jBlock);
                final int jWidth2 = jWidth  + jWidth;
                final int jWidth3 = jWidth2 + jWidth;
                final int jWidth4 = jWidth3 + jWidth;

                // select current block
                final double[] outBlock = out.blocks[blockIndex];

                // perform multiplication on current block
                for (int kBlock = 0; kBlock < blockColumns; ++kBlock) {
                    final int kWidth = blockWidth(kBlock);
                    final double[] tBlock = blocks[iBlock * blockColumns + kBlock];
                    final double[] mBlock = m.blocks[kBlock * m.blockColumns + jBlock];
                    int k = 0;
                    for (int p = pStart; p < pEnd; ++p) {
                        final int lStart = (p - pStart) * kWidth;
                        final int lEnd = lStart + kWidth;
                        for (int nStart = 0; nStart < jWidth; ++nStart) {
                            double sum = 0;
                            int l = lStart;
                            int n = nStart;
                            while (l < lEnd - 3) {
                                sum += tBlock[l] * mBlock[n] +
                                       tBlock[l + 1] * mBlock[n + jWidth] +
                                       tBlock[l + 2] * mBlock[n + jWidth2] +
                                       tBlock[l + 3] * mBlock[n + jWidth3];
                                l += 4;
                                n += jWidth4;
                            }
                            while (l < lEnd) {
                                sum += tBlock[l++] * mBlock[n];
                                n += jWidth;
                            }
                            outBlock[k] += sum;
                            ++k;
                        }
                    }
                }
                // go to next block
                ++blockIndex;
            }
        }

        return out;
    }

    
    @Override
    public double[][] getData() {
        final double[][] data = new double[getRowDimension()][getColumnDimension()];
        final int lastColumns = columns - (blockColumns - 1) * BLOCK_SIZE;

        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
            int regularPos = 0;
            int lastPos = 0;
            for (int p = pStart; p < pEnd; ++p) {
                final double[] dataP = data[p];
                int blockIndex = iBlock * blockColumns;
                int dataPos = 0;
                for (int jBlock = 0; jBlock < blockColumns - 1; ++jBlock) {
                    System.arraycopy(blocks[blockIndex++], regularPos, dataP, dataPos, BLOCK_SIZE);
                    dataPos += BLOCK_SIZE;
                }
                System.arraycopy(blocks[blockIndex], lastPos, dataP, dataPos, lastColumns);
                regularPos += BLOCK_SIZE;
                lastPos    += lastColumns;
            }
        }

        return data;
    }

    
    @Override
    public double getNorm() {
        final double[] colSums = new double[BLOCK_SIZE];
        double maxColSum = 0;
        for (int jBlock = 0; jBlock < blockColumns; jBlock++) {
            final int jWidth = blockWidth(jBlock);
            Arrays.fill(colSums, 0, jWidth, 0.0);
            for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
                final int iHeight = blockHeight(iBlock);
                final double[] block = blocks[iBlock * blockColumns + jBlock];
                for (int j = 0; j < jWidth; ++j) {
                    double sum = 0;
                    for (int i = 0; i < iHeight; ++i) {
                        sum += FastMath.abs(block[i * jWidth + j]);
                    }
                    colSums[j] += sum;
                }
            }
            for (int j = 0; j < jWidth; ++j) {
                maxColSum = FastMath.max(maxColSum, colSums[j]);
            }
        }
        return maxColSum;
    }

    
    @Override
    public double getFrobeniusNorm() {
        double sum2 = 0;
        for (int blockIndex = 0; blockIndex < blocks.length; ++blockIndex) {
            for (final double entry : blocks[blockIndex]) {
                sum2 += entry * entry;
            }
        }
        return FastMath.sqrt(sum2);
    }

    
    @Override
    public BlockRealMatrix getSubMatrix(final int startRow, final int endRow,
                                        final int startColumn,
                                        final int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException {
        // safety checks
        MatrixUtils.checkSubMatrixIndex(this, startRow, endRow, startColumn, endColumn);

        // create the output matrix
        final BlockRealMatrix out =
            new BlockRealMatrix(endRow - startRow + 1, endColumn - startColumn + 1);

        // compute blocks shifts
        final int blockStartRow = startRow / BLOCK_SIZE;
        final int rowsShift = startRow % BLOCK_SIZE;
        final int blockStartColumn = startColumn / BLOCK_SIZE;
        final int columnsShift = startColumn % BLOCK_SIZE;

        // perform extraction block-wise, to ensure good cache behavior
        int pBlock = blockStartRow;
        for (int iBlock = 0; iBlock < out.blockRows; ++iBlock) {
            final int iHeight = out.blockHeight(iBlock);
            int qBlock = blockStartColumn;
            for (int jBlock = 0; jBlock < out.blockColumns; ++jBlock) {
                final int jWidth = out.blockWidth(jBlock);

                // handle one block of the output matrix
                final int outIndex = iBlock * out.blockColumns + jBlock;
                final double[] outBlock = out.blocks[outIndex];
                final int index = pBlock * blockColumns + qBlock;
                final int width = blockWidth(qBlock);

                final int heightExcess = iHeight + rowsShift - BLOCK_SIZE;
                final int widthExcess = jWidth + columnsShift - BLOCK_SIZE;
                if (heightExcess > 0) {
                    // the submatrix block spans on two blocks rows from the original matrix
                    if (widthExcess > 0) {
                        // the submatrix block spans on two blocks columns from the original matrix
                        final int width2 = blockWidth(qBlock + 1);
                        copyBlockPart(blocks[index], width,
                                      rowsShift, BLOCK_SIZE,
                                      columnsShift, BLOCK_SIZE,
                                      outBlock, jWidth, 0, 0);
                        copyBlockPart(blocks[index + 1], width2,
                                      rowsShift, BLOCK_SIZE,
                                      0, widthExcess,
                                      outBlock, jWidth, 0, jWidth - widthExcess);
                        copyBlockPart(blocks[index + blockColumns], width,
                                      0, heightExcess,
                                      columnsShift, BLOCK_SIZE,
                                      outBlock, jWidth, iHeight - heightExcess, 0);
                        copyBlockPart(blocks[index + blockColumns + 1], width2,
                                      0, heightExcess,
                                      0, widthExcess,
                                      outBlock, jWidth, iHeight - heightExcess, jWidth - widthExcess);
                    } else {
                        // the submatrix block spans on one block column from the original matrix
                        copyBlockPart(blocks[index], width,
                                      rowsShift, BLOCK_SIZE,
                                      columnsShift, jWidth + columnsShift,
                                      outBlock, jWidth, 0, 0);
                        copyBlockPart(blocks[index + blockColumns], width,
                                      0, heightExcess,
                                      columnsShift, jWidth + columnsShift,
                                      outBlock, jWidth, iHeight - heightExcess, 0);
                    }
                } else {
                    // the submatrix block spans on one block row from the original matrix
                    if (widthExcess > 0) {
                        // the submatrix block spans on two blocks columns from the original matrix
                        final int width2 = blockWidth(qBlock + 1);
                        copyBlockPart(blocks[index], width,
                                      rowsShift, iHeight + rowsShift,
                                      columnsShift, BLOCK_SIZE,
                                      outBlock, jWidth, 0, 0);
                        copyBlockPart(blocks[index + 1], width2,
                                      rowsShift, iHeight + rowsShift,
                                      0, widthExcess,
                                      outBlock, jWidth, 0, jWidth - widthExcess);
                    } else {
                        // the submatrix block spans on one block column from the original matrix
                        copyBlockPart(blocks[index], width,
                                      rowsShift, iHeight + rowsShift,
                                      columnsShift, jWidth + columnsShift,
                                      outBlock, jWidth, 0, 0);
                    }
               }
                ++qBlock;
            }
            ++pBlock;
        }

        return out;
    }

    
    private void copyBlockPart(final double[] srcBlock, final int srcWidth,
                               final int srcStartRow, final int srcEndRow,
                               final int srcStartColumn, final int srcEndColumn,
                               final double[] dstBlock, final int dstWidth,
                               final int dstStartRow, final int dstStartColumn) {
        final int length = srcEndColumn - srcStartColumn;
        int srcPos = srcStartRow * srcWidth + srcStartColumn;
        int dstPos = dstStartRow * dstWidth + dstStartColumn;
        for (int srcRow = srcStartRow; srcRow < srcEndRow; ++srcRow) {
            System.arraycopy(srcBlock, srcPos, dstBlock, dstPos, length);
            srcPos += srcWidth;
            dstPos += dstWidth;
        }
    }

    
    @Override
    public void setSubMatrix(final double[][] subMatrix, final int row,
                             final int column)
        throws OutOfRangeException, NoDataException, NullArgumentException,
        DimensionMismatchException {
        // safety checks
        MathUtils.checkNotNull(subMatrix);
        final int refLength = subMatrix[0].length;
        if (refLength == 0) {
            throw new NoDataException(LocalizedFormats.AT_LEAST_ONE_COLUMN);
        }
        final int endRow = row + subMatrix.length - 1;
        final int endColumn = column + refLength - 1;
        MatrixUtils.checkSubMatrixIndex(this, row, endRow, column, endColumn);
        for (final double[] subRow : subMatrix) {
            if (subRow.length != refLength) {
                throw new DimensionMismatchException(refLength, subRow.length);
            }
        }

        // compute blocks bounds
        final int blockStartRow = row / BLOCK_SIZE;
        final int blockEndRow = (endRow + BLOCK_SIZE) / BLOCK_SIZE;
        final int blockStartColumn = column / BLOCK_SIZE;
        final int blockEndColumn = (endColumn + BLOCK_SIZE) / BLOCK_SIZE;

        // perform copy block-wise, to ensure good cache behavior
        for (int iBlock = blockStartRow; iBlock < blockEndRow; ++iBlock) {
            final int iHeight = blockHeight(iBlock);
            final int firstRow = iBlock * BLOCK_SIZE;
            final int iStart = FastMath.max(row,    firstRow);
            final int iEnd = FastMath.min(endRow + 1, firstRow + iHeight);

            for (int jBlock = blockStartColumn; jBlock < blockEndColumn; ++jBlock) {
                final int jWidth = blockWidth(jBlock);
                final int firstColumn = jBlock * BLOCK_SIZE;
                final int jStart = FastMath.max(column,    firstColumn);
                final int jEnd = FastMath.min(endColumn + 1, firstColumn + jWidth);
                final int jLength = jEnd - jStart;

                // handle one block, row by row
                final double[] block = blocks[iBlock * blockColumns + jBlock];
                for (int i = iStart; i < iEnd; ++i) {
                    System.arraycopy(subMatrix[i - row], jStart - column,
                                     block, (i - firstRow) * jWidth + (jStart - firstColumn),
                                     jLength);
                }

            }
        }
    }

    
    @Override
    public BlockRealMatrix getRowMatrix(final int row)
        throws OutOfRangeException {
        MatrixUtils.checkRowIndex(this, row);
        final BlockRealMatrix out = new BlockRealMatrix(1, columns);

        // perform copy block-wise, to ensure good cache behavior
        final int iBlock = row / BLOCK_SIZE;
        final int iRow = row - iBlock * BLOCK_SIZE;
        int outBlockIndex = 0;
        int outIndex = 0;
        double[] outBlock = out.blocks[outBlockIndex];
        for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
            final int jWidth = blockWidth(jBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            final int available = outBlock.length - outIndex;
            if (jWidth > available) {
                System.arraycopy(block, iRow * jWidth, outBlock, outIndex, available);
                outBlock = out.blocks[++outBlockIndex];
                System.arraycopy(block, iRow * jWidth, outBlock, 0, jWidth - available);
                outIndex = jWidth - available;
            } else {
                System.arraycopy(block, iRow * jWidth, outBlock, outIndex, jWidth);
                outIndex += jWidth;
            }
        }

        return out;
    }

    
    @Override
    public void setRowMatrix(final int row, final RealMatrix matrix)
        throws OutOfRangeException, MatrixDimensionMismatchException {
        try {
            setRowMatrix(row, (BlockRealMatrix) matrix);
        } catch (ClassCastException cce) {
            super.setRowMatrix(row, matrix);
        }
    }

    
    public void setRowMatrix(final int row, final BlockRealMatrix matrix)
        throws OutOfRangeException, MatrixDimensionMismatchException {
        MatrixUtils.checkRowIndex(this, row);
        final int nCols = getColumnDimension();
        if ((matrix.getRowDimension() != 1) ||
            (matrix.getColumnDimension() != nCols)) {
            throw new MatrixDimensionMismatchException(matrix.getRowDimension(),
                                                       matrix.getColumnDimension(),
                                                       1, nCols);
        }

        // perform copy block-wise, to ensure good cache behavior
        final int iBlock = row / BLOCK_SIZE;
        final int iRow = row - iBlock * BLOCK_SIZE;
        int mBlockIndex = 0;
        int mIndex = 0;
        double[] mBlock = matrix.blocks[mBlockIndex];
        for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
            final int jWidth = blockWidth(jBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            final int available  = mBlock.length - mIndex;
            if (jWidth > available) {
                System.arraycopy(mBlock, mIndex, block, iRow * jWidth, available);
                mBlock = matrix.blocks[++mBlockIndex];
                System.arraycopy(mBlock, 0, block, iRow * jWidth, jWidth - available);
                mIndex = jWidth - available;
            } else {
                System.arraycopy(mBlock, mIndex, block, iRow * jWidth, jWidth);
                mIndex += jWidth;
           }
        }
    }

    
    @Override
    public BlockRealMatrix getColumnMatrix(final int column)
        throws OutOfRangeException {
        MatrixUtils.checkColumnIndex(this, column);
        final BlockRealMatrix out = new BlockRealMatrix(rows, 1);

        // perform copy block-wise, to ensure good cache behavior
        final int jBlock = column / BLOCK_SIZE;
        final int jColumn = column - jBlock * BLOCK_SIZE;
        final int jWidth = blockWidth(jBlock);
        int outBlockIndex = 0;
        int outIndex = 0;
        double[] outBlock = out.blocks[outBlockIndex];
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int iHeight = blockHeight(iBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            for (int i = 0; i < iHeight; ++i) {
                if (outIndex >= outBlock.length) {
                    outBlock = out.blocks[++outBlockIndex];
                    outIndex = 0;
                }
                outBlock[outIndex++] = block[i * jWidth + jColumn];
            }
        }

        return out;
    }

    
    @Override
    public void setColumnMatrix(final int column, final RealMatrix matrix)
        throws OutOfRangeException, MatrixDimensionMismatchException {
        try {
            setColumnMatrix(column, (BlockRealMatrix) matrix);
        } catch (ClassCastException cce) {
            super.setColumnMatrix(column, matrix);
        }
    }

    
    void setColumnMatrix(final int column, final BlockRealMatrix matrix)
        throws OutOfRangeException, MatrixDimensionMismatchException {
        MatrixUtils.checkColumnIndex(this, column);
        final int nRows = getRowDimension();
        if ((matrix.getRowDimension() != nRows) ||
            (matrix.getColumnDimension() != 1)) {
            throw new MatrixDimensionMismatchException(matrix.getRowDimension(),
                                                       matrix.getColumnDimension(),
                                                       nRows, 1);
        }

        // perform copy block-wise, to ensure good cache behavior
        final int jBlock = column / BLOCK_SIZE;
        final int jColumn = column - jBlock * BLOCK_SIZE;
        final int jWidth = blockWidth(jBlock);
        int mBlockIndex = 0;
        int mIndex = 0;
        double[] mBlock = matrix.blocks[mBlockIndex];
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int iHeight = blockHeight(iBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            for (int i = 0; i < iHeight; ++i) {
                if (mIndex >= mBlock.length) {
                    mBlock = matrix.blocks[++mBlockIndex];
                    mIndex = 0;
                }
                block[i * jWidth + jColumn] = mBlock[mIndex++];
            }
        }
    }

    
    @Override
    public RealVector getRowVector(final int row)
        throws OutOfRangeException {
        MatrixUtils.checkRowIndex(this, row);
        final double[] outData = new double[columns];

        // perform copy block-wise, to ensure good cache behavior
        final int iBlock = row / BLOCK_SIZE;
        final int iRow = row - iBlock * BLOCK_SIZE;
        int outIndex = 0;
        for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
            final int jWidth = blockWidth(jBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            System.arraycopy(block, iRow * jWidth, outData, outIndex, jWidth);
            outIndex += jWidth;
        }

        return new ArrayRealVector(outData, false);
    }

    
    @Override
    public void setRowVector(final int row, final RealVector vector)
        throws OutOfRangeException, MatrixDimensionMismatchException {
        try {
            setRow(row, ((ArrayRealVector) vector).getDataRef());
        } catch (ClassCastException cce) {
            super.setRowVector(row, vector);
        }
    }

    
    @Override
    public RealVector getColumnVector(final int column)
        throws OutOfRangeException {
        MatrixUtils.checkColumnIndex(this, column);
        final double[] outData = new double[rows];

        // perform copy block-wise, to ensure good cache behavior
        final int jBlock = column / BLOCK_SIZE;
        final int jColumn = column - jBlock * BLOCK_SIZE;
        final int jWidth = blockWidth(jBlock);
        int outIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int iHeight = blockHeight(iBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            for (int i = 0; i < iHeight; ++i) {
                outData[outIndex++] = block[i * jWidth + jColumn];
            }
        }

        return new ArrayRealVector(outData, false);
    }

    
    @Override
    public void setColumnVector(final int column, final RealVector vector)
        throws OutOfRangeException, MatrixDimensionMismatchException {
        try {
            setColumn(column, ((ArrayRealVector) vector).getDataRef());
        } catch (ClassCastException cce) {
            super.setColumnVector(column, vector);
        }
    }

    
    @Override
    public double[] getRow(final int row) throws OutOfRangeException {
        MatrixUtils.checkRowIndex(this, row);
        final double[] out = new double[columns];

        // perform copy block-wise, to ensure good cache behavior
        final int iBlock = row / BLOCK_SIZE;
        final int iRow = row - iBlock * BLOCK_SIZE;
        int outIndex = 0;
        for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
            final int jWidth     = blockWidth(jBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            System.arraycopy(block, iRow * jWidth, out, outIndex, jWidth);
            outIndex += jWidth;
        }

        return out;
    }

    
    @Override
    public void setRow(final int row, final double[] array)
        throws OutOfRangeException, MatrixDimensionMismatchException {
        MatrixUtils.checkRowIndex(this, row);
        final int nCols = getColumnDimension();
        if (array.length != nCols) {
            throw new MatrixDimensionMismatchException(1, array.length, 1, nCols);
        }

        // perform copy block-wise, to ensure good cache behavior
        final int iBlock = row / BLOCK_SIZE;
        final int iRow = row - iBlock * BLOCK_SIZE;
        int outIndex = 0;
        for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
            final int jWidth     = blockWidth(jBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            System.arraycopy(array, outIndex, block, iRow * jWidth, jWidth);
            outIndex += jWidth;
        }
    }

    
    @Override
    public double[] getColumn(final int column) throws OutOfRangeException {
        MatrixUtils.checkColumnIndex(this, column);
        final double[] out = new double[rows];

        // perform copy block-wise, to ensure good cache behavior
        final int jBlock  = column / BLOCK_SIZE;
        final int jColumn = column - jBlock * BLOCK_SIZE;
        final int jWidth  = blockWidth(jBlock);
        int outIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int iHeight = blockHeight(iBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            for (int i = 0; i < iHeight; ++i) {
                out[outIndex++] = block[i * jWidth + jColumn];
            }
        }

        return out;
    }

    
    @Override
    public void setColumn(final int column, final double[] array)
        throws OutOfRangeException, MatrixDimensionMismatchException {
        MatrixUtils.checkColumnIndex(this, column);
        final int nRows = getRowDimension();
        if (array.length != nRows) {
            throw new MatrixDimensionMismatchException(array.length, 1, nRows, 1);
        }

        // perform copy block-wise, to ensure good cache behavior
        final int jBlock  = column / BLOCK_SIZE;
        final int jColumn = column - jBlock * BLOCK_SIZE;
        final int jWidth = blockWidth(jBlock);
        int outIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int iHeight = blockHeight(iBlock);
            final double[] block = blocks[iBlock * blockColumns + jBlock];
            for (int i = 0; i < iHeight; ++i) {
                block[i * jWidth + jColumn] = array[outIndex++];
            }
        }
    }

    
    @Override
    public double getEntry(final int row, final int column)
        throws OutOfRangeException {
        MatrixUtils.checkMatrixIndex(this, row, column);
        final int iBlock = row / BLOCK_SIZE;
        final int jBlock = column / BLOCK_SIZE;
        final int k = (row - iBlock * BLOCK_SIZE) * blockWidth(jBlock) +
            (column - jBlock * BLOCK_SIZE);
        return blocks[iBlock * blockColumns + jBlock][k];
    }

    
    @Override
    public void setEntry(final int row, final int column, final double value)
        throws OutOfRangeException {
        MatrixUtils.checkMatrixIndex(this, row, column);
        final int iBlock = row / BLOCK_SIZE;
        final int jBlock = column / BLOCK_SIZE;
        final int k = (row - iBlock * BLOCK_SIZE) * blockWidth(jBlock) +
            (column - jBlock * BLOCK_SIZE);
        blocks[iBlock * blockColumns + jBlock][k] = value;
    }

    
    @Override
    public void addToEntry(final int row, final int column,
                           final double increment)
        throws OutOfRangeException {
        MatrixUtils.checkMatrixIndex(this, row, column);
        final int iBlock = row    / BLOCK_SIZE;
        final int jBlock = column / BLOCK_SIZE;
        final int k = (row    - iBlock * BLOCK_SIZE) * blockWidth(jBlock) +
            (column - jBlock * BLOCK_SIZE);
        blocks[iBlock * blockColumns + jBlock][k] += increment;
    }

    
    @Override
    public void multiplyEntry(final int row, final int column,
                              final double factor)
        throws OutOfRangeException {
        MatrixUtils.checkMatrixIndex(this, row, column);
        final int iBlock = row / BLOCK_SIZE;
        final int jBlock = column / BLOCK_SIZE;
        final int k = (row - iBlock * BLOCK_SIZE) * blockWidth(jBlock) +
            (column - jBlock * BLOCK_SIZE);
        blocks[iBlock * blockColumns + jBlock][k] *= factor;
    }

    
    @Override
    public BlockRealMatrix transpose() {
        final int nRows = getRowDimension();
        final int nCols = getColumnDimension();
        final BlockRealMatrix out = new BlockRealMatrix(nCols, nRows);

        // perform transpose block-wise, to ensure good cache behavior
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockColumns; ++iBlock) {
            for (int jBlock = 0; jBlock < blockRows; ++jBlock) {
                // transpose current block
                final double[] outBlock = out.blocks[blockIndex];
                final double[] tBlock = blocks[jBlock * blockColumns + iBlock];
                final int pStart = iBlock * BLOCK_SIZE;
                final int pEnd = FastMath.min(pStart + BLOCK_SIZE, columns);
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = FastMath.min(qStart + BLOCK_SIZE, rows);
                int k = 0;
                for (int p = pStart; p < pEnd; ++p) {
                    final int lInc = pEnd - pStart;
                    int l = p - pStart;
                    for (int q = qStart; q < qEnd; ++q) {
                        outBlock[k] = tBlock[l];
                        ++k;
                        l+= lInc;
                    }
                }
                // go to next block
                ++blockIndex;
            }
        }

        return out;
    }

    
    @Override
    public int getRowDimension() {
        return rows;
    }

    
    @Override
    public int getColumnDimension() {
        return columns;
    }

    
    @Override
    public double[] operate(final double[] v)
        throws DimensionMismatchException {
        if (v.length != columns) {
            throw new DimensionMismatchException(v.length, columns);
        }
        final double[] out = new double[rows];

        // perform multiplication block-wise, to ensure good cache behavior
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final double[] block  = blocks[iBlock * blockColumns + jBlock];
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
                int k = 0;
                for (int p = pStart; p < pEnd; ++p) {
                    double sum = 0;
                    int q = qStart;
                    while (q < qEnd - 3) {
                        sum += block[k]     * v[q]     +
                               block[k + 1] * v[q + 1] +
                               block[k + 2] * v[q + 2] +
                               block[k + 3] * v[q + 3];
                        k += 4;
                        q += 4;
                    }
                    while (q < qEnd) {
                        sum += block[k++] * v[q++];
                    }
                    out[p] += sum;
                }
            }
        }

        return out;
    }

    
    @Override
    public double[] preMultiply(final double[] v)
        throws DimensionMismatchException {
        if (v.length != rows) {
            throw new DimensionMismatchException(v.length, rows);
        }
        final double[] out = new double[columns];

        // perform multiplication block-wise, to ensure good cache behavior
        for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
            final int jWidth  = blockWidth(jBlock);
            final int jWidth2 = jWidth  + jWidth;
            final int jWidth3 = jWidth2 + jWidth;
            final int jWidth4 = jWidth3 + jWidth;
            final int qStart = jBlock * BLOCK_SIZE;
            final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
            for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
                final double[] block  = blocks[iBlock * blockColumns + jBlock];
                final int pStart = iBlock * BLOCK_SIZE;
                final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
                for (int q = qStart; q < qEnd; ++q) {
                    int k = q - qStart;
                    double sum = 0;
                    int p = pStart;
                    while (p < pEnd - 3) {
                        sum += block[k]           * v[p]     +
                               block[k + jWidth]  * v[p + 1] +
                               block[k + jWidth2] * v[p + 2] +
                               block[k + jWidth3] * v[p + 3];
                        k += jWidth4;
                        p += 4;
                    }
                    while (p < pEnd) {
                        sum += block[k] * v[p++];
                        k += jWidth;
                    }
                    out[q] += sum;
                }
            }
        }

        return out;
    }

    
    @Override
    public double walkInRowOrder(final RealMatrixChangingVisitor visitor) {
        visitor.start(rows, columns, 0, rows - 1, 0, columns - 1);
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
                    final double[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        block[k] = visitor.visit(p, q, block[k]);
                        ++k;
                    }
                }
             }
        }
        return visitor.end();
    }

    
    @Override
    public double walkInRowOrder(final RealMatrixPreservingVisitor visitor) {
        visitor.start(rows, columns, 0, rows - 1, 0, columns - 1);
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
                    final double[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        visitor.visit(p, q, block[k]);
                        ++k;
                    }
                }
             }
        }
        return visitor.end();
    }

    
    @Override
    public double walkInRowOrder(final RealMatrixChangingVisitor visitor,
                                 final int startRow, final int endRow,
                                 final int startColumn, final int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException {
        MatrixUtils.checkSubMatrixIndex(this, startRow, endRow, startColumn, endColumn);
        visitor.start(rows, columns, startRow, endRow, startColumn, endColumn);
        for (int iBlock = startRow / BLOCK_SIZE; iBlock < 1 + endRow / BLOCK_SIZE; ++iBlock) {
            final int p0 = iBlock * BLOCK_SIZE;
            final int pStart = FastMath.max(startRow, p0);
            final int pEnd = FastMath.min((iBlock + 1) * BLOCK_SIZE, 1 + endRow);
            for (int p = pStart; p < pEnd; ++p) {
                for (int jBlock = startColumn / BLOCK_SIZE; jBlock < 1 + endColumn / BLOCK_SIZE; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int q0 = jBlock * BLOCK_SIZE;
                    final int qStart = FastMath.max(startColumn, q0);
                    final int qEnd = FastMath.min((jBlock + 1) * BLOCK_SIZE, 1 + endColumn);
                    final double[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - p0) * jWidth + qStart - q0;
                    for (int q = qStart; q < qEnd; ++q) {
                        block[k] = visitor.visit(p, q, block[k]);
                        ++k;
                    }
                }
             }
        }
        return visitor.end();
    }

    
    @Override
    public double walkInRowOrder(final RealMatrixPreservingVisitor visitor,
                                 final int startRow, final int endRow,
                                 final int startColumn, final int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException {
        MatrixUtils.checkSubMatrixIndex(this, startRow, endRow, startColumn, endColumn);
        visitor.start(rows, columns, startRow, endRow, startColumn, endColumn);
        for (int iBlock = startRow / BLOCK_SIZE; iBlock < 1 + endRow / BLOCK_SIZE; ++iBlock) {
            final int p0 = iBlock * BLOCK_SIZE;
            final int pStart = FastMath.max(startRow, p0);
            final int pEnd = FastMath.min((iBlock + 1) * BLOCK_SIZE, 1 + endRow);
            for (int p = pStart; p < pEnd; ++p) {
                for (int jBlock = startColumn / BLOCK_SIZE; jBlock < 1 + endColumn / BLOCK_SIZE; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int q0 = jBlock * BLOCK_SIZE;
                    final int qStart = FastMath.max(startColumn, q0);
                    final int qEnd = FastMath.min((jBlock + 1) * BLOCK_SIZE, 1 + endColumn);
                    final double[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - p0) * jWidth + qStart - q0;
                    for (int q = qStart; q < qEnd; ++q) {
                        visitor.visit(p, q, block[k]);
                        ++k;
                    }
                }
             }
        }
        return visitor.end();
    }

    
    @Override
    public double walkInOptimizedOrder(final RealMatrixChangingVisitor visitor) {
        visitor.start(rows, columns, 0, rows - 1, 0, columns - 1);
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
                final double[] block = blocks[blockIndex];
                int k = 0;
                for (int p = pStart; p < pEnd; ++p) {
                    for (int q = qStart; q < qEnd; ++q) {
                        block[k] = visitor.visit(p, q, block[k]);
                        ++k;
                    }
                }
                ++blockIndex;
            }
        }
        return visitor.end();
    }

    
    @Override
    public double walkInOptimizedOrder(final RealMatrixPreservingVisitor visitor) {
        visitor.start(rows, columns, 0, rows - 1, 0, columns - 1);
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = FastMath.min(pStart + BLOCK_SIZE, rows);
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = FastMath.min(qStart + BLOCK_SIZE, columns);
                final double[] block = blocks[blockIndex];
                int k = 0;
                for (int p = pStart; p < pEnd; ++p) {
                    for (int q = qStart; q < qEnd; ++q) {
                        visitor.visit(p, q, block[k]);
                        ++k;
                    }
                }
                ++blockIndex;
            }
        }
        return visitor.end();
    }

    
    @Override
    public double walkInOptimizedOrder(final RealMatrixChangingVisitor visitor,
                                       final int startRow, final int endRow,
                                       final int startColumn,
                                       final int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException {
        MatrixUtils.checkSubMatrixIndex(this, startRow, endRow, startColumn, endColumn);
        visitor.start(rows, columns, startRow, endRow, startColumn, endColumn);
        for (int iBlock = startRow / BLOCK_SIZE; iBlock < 1 + endRow / BLOCK_SIZE; ++iBlock) {
            final int p0 = iBlock * BLOCK_SIZE;
            final int pStart = FastMath.max(startRow, p0);
            final int pEnd = FastMath.min((iBlock + 1) * BLOCK_SIZE, 1 + endRow);
            for (int jBlock = startColumn / BLOCK_SIZE; jBlock < 1 + endColumn / BLOCK_SIZE; ++jBlock) {
                final int jWidth = blockWidth(jBlock);
                final int q0 = jBlock * BLOCK_SIZE;
                final int qStart = FastMath.max(startColumn, q0);
                final int qEnd = FastMath.min((jBlock + 1) * BLOCK_SIZE, 1 + endColumn);
                final double[] block = blocks[iBlock * blockColumns + jBlock];
                for (int p = pStart; p < pEnd; ++p) {
                    int k = (p - p0) * jWidth + qStart - q0;
                    for (int q = qStart; q < qEnd; ++q) {
                        block[k] = visitor.visit(p, q, block[k]);
                        ++k;
                    }
                }
            }
        }
        return visitor.end();
    }

    
    @Override
    public double walkInOptimizedOrder(final RealMatrixPreservingVisitor visitor,
                                       final int startRow, final int endRow,
                                       final int startColumn,
                                       final int endColumn)
        throws OutOfRangeException, NumberIsTooSmallException {
        MatrixUtils.checkSubMatrixIndex(this, startRow, endRow, startColumn, endColumn);
        visitor.start(rows, columns, startRow, endRow, startColumn, endColumn);
        for (int iBlock = startRow / BLOCK_SIZE; iBlock < 1 + endRow / BLOCK_SIZE; ++iBlock) {
            final int p0 = iBlock * BLOCK_SIZE;
            final int pStart = FastMath.max(startRow, p0);
            final int pEnd = FastMath.min((iBlock + 1) * BLOCK_SIZE, 1 + endRow);
            for (int jBlock = startColumn / BLOCK_SIZE; jBlock < 1 + endColumn / BLOCK_SIZE; ++jBlock) {
                final int jWidth = blockWidth(jBlock);
                final int q0 = jBlock * BLOCK_SIZE;
                final int qStart = FastMath.max(startColumn, q0);
                final int qEnd = FastMath.min((jBlock + 1) * BLOCK_SIZE, 1 + endColumn);
                final double[] block = blocks[iBlock * blockColumns + jBlock];
                for (int p = pStart; p < pEnd; ++p) {
                    int k = (p - p0) * jWidth + qStart - q0;
                    for (int q = qStart; q < qEnd; ++q) {
                        visitor.visit(p, q, block[k]);
                        ++k;
                    }
                }
            }
        }
        return visitor.end();
    }

    
    private int blockHeight(final int blockRow) {
        return (blockRow == blockRows - 1) ? rows - blockRow * BLOCK_SIZE : BLOCK_SIZE;
    }

    
    private int blockWidth(final int blockColumn) {
        return (blockColumn == blockColumns - 1) ? columns - blockColumn * BLOCK_SIZE : BLOCK_SIZE;
    }
}
