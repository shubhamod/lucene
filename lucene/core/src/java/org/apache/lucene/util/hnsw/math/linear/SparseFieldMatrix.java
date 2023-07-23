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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.util.OpenIntToFieldHashMap;


public class SparseFieldMatrix<T extends FieldElement<T>> extends AbstractFieldMatrix<T> {

    
    private final OpenIntToFieldHashMap<T> entries;
    
    private final int rows;
    
    private final int columns;

    
    public SparseFieldMatrix(final Field<T> field) {
        super(field);
        rows = 0;
        columns= 0;
        entries = new OpenIntToFieldHashMap<T>(field);
    }

    
    public SparseFieldMatrix(final Field<T> field,
                             final int rowDimension, final int columnDimension) {
        super(field, rowDimension, columnDimension);
        this.rows = rowDimension;
        this.columns = columnDimension;
        entries = new OpenIntToFieldHashMap<T>(field);
    }

    
    public SparseFieldMatrix(SparseFieldMatrix<T> other) {
        super(other.getField(), other.getRowDimension(), other.getColumnDimension());
        rows = other.getRowDimension();
        columns = other.getColumnDimension();
        entries = new OpenIntToFieldHashMap<T>(other.entries);
    }

    
    public SparseFieldMatrix(FieldMatrix<T> other){
        super(other.getField(), other.getRowDimension(), other.getColumnDimension());
        rows = other.getRowDimension();
        columns = other.getColumnDimension();
        entries = new OpenIntToFieldHashMap<T>(getField());
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                setEntry(i, j, other.getEntry(i, j));
            }
        }
    }

    
    @Override
    public void addToEntry(int row, int column, T increment) {
        checkRowIndex(row);
        checkColumnIndex(column);
        final int key = computeKey(row, column);
        final T value = entries.get(key).add(increment);
        if (getField().getZero().equals(value)) {
            entries.remove(key);
        } else {
            entries.put(key, value);
        }
    }

    
    @Override
    public FieldMatrix<T> copy() {
        return new SparseFieldMatrix<T>(this);
    }

    
    @Override
    public FieldMatrix<T> createMatrix(int rowDimension, int columnDimension) {
        return new SparseFieldMatrix<T>(getField(), rowDimension, columnDimension);
    }

    
    @Override
    public int getColumnDimension() {
        return columns;
    }

    
    @Override
    public T getEntry(int row, int column) {
        checkRowIndex(row);
        checkColumnIndex(column);
        return entries.get(computeKey(row, column));
    }

    
    @Override
    public int getRowDimension() {
        return rows;
    }

    
    @Override
    public void multiplyEntry(int row, int column, T factor) {
        checkRowIndex(row);
        checkColumnIndex(column);
        final int key = computeKey(row, column);
        final T value = entries.get(key).multiply(factor);
        if (getField().getZero().equals(value)) {
            entries.remove(key);
        } else {
            entries.put(key, value);
        }

    }

    
    @Override
    public void setEntry(int row, int column, T value) {
        checkRowIndex(row);
        checkColumnIndex(column);
        if (getField().getZero().equals(value)) {
            entries.remove(computeKey(row, column));
        } else {
            entries.put(computeKey(row, column), value);
        }
    }

    
    private int computeKey(int row, int column) {
        return row * columns + column;
    }
}
