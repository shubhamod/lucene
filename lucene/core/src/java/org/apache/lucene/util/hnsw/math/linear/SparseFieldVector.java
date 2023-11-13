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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.OpenIntToFieldHashMap;


public class SparseFieldVector<T extends FieldElement<T>> implements FieldVector<T>, Serializable {
    
    private static final long serialVersionUID = 7841233292190413362L;
    
    private final Field<T> field;
    
    private final OpenIntToFieldHashMap<T> entries;
    
    private final int virtualSize;

    
    public SparseFieldVector(Field<T> field) {
        this(field, 0);
    }


    
    public SparseFieldVector(Field<T> field, int dimension) {
        this.field = field;
        virtualSize = dimension;
        entries = new OpenIntToFieldHashMap<T>(field);
    }

    
    protected SparseFieldVector(SparseFieldVector<T> v, int resize) {
        field = v.field;
        virtualSize = v.getDimension() + resize;
        entries = new OpenIntToFieldHashMap<T>(v.entries);
    }


    
    public SparseFieldVector(Field<T> field, int dimension, int expectedSize) {
        this.field = field;
        virtualSize = dimension;
        entries = new OpenIntToFieldHashMap<T>(field,expectedSize);
    }

    
    public SparseFieldVector(Field<T> field, T[] values) throws NullArgumentException {
        MathUtils.checkNotNull(values);
        this.field = field;
        virtualSize = values.length;
        entries = new OpenIntToFieldHashMap<T>(field);
        for (int key = 0; key < values.length; key++) {
            T value = values[key];
            entries.put(key, value);
        }
    }

    
    public SparseFieldVector(SparseFieldVector<T> v) {
        field = v.field;
        virtualSize = v.getDimension();
        entries = new OpenIntToFieldHashMap<T>(v.getEntries());
    }

    
    private OpenIntToFieldHashMap<T> getEntries() {
        return entries;
    }

    
    public FieldVector<T> add(SparseFieldVector<T> v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        SparseFieldVector<T> res = (SparseFieldVector<T>)copy();
        OpenIntToFieldHashMap<T>.Iterator iter = v.getEntries().iterator();
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            T value = iter.value();
            if (entries.containsKey(key)) {
                res.setEntry(key, entries.get(key).add(value));
            } else {
                res.setEntry(key, value);
            }
        }
        return res;

    }

    
    public FieldVector<T> append(SparseFieldVector<T> v) {
        SparseFieldVector<T> res = new SparseFieldVector<T>(this, v.getDimension());
        OpenIntToFieldHashMap<T>.Iterator iter = v.entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            res.setEntry(iter.key() + virtualSize, iter.value());
        }
        return res;
    }

    
    public FieldVector<T> append(FieldVector<T> v) {
        if (v instanceof SparseFieldVector<?>) {
            return append((SparseFieldVector<T>) v);
        } else {
            final int n = v.getDimension();
            FieldVector<T> res = new SparseFieldVector<T>(this, n);
            for (int i = 0; i < n; i++) {
                res.setEntry(i + virtualSize, v.getEntry(i));
            }
            return res;
        }
    }

    
    public FieldVector<T> append(T d) throws NullArgumentException {
        MathUtils.checkNotNull(d);
        FieldVector<T> res = new SparseFieldVector<T>(this, 1);
        res.setEntry(virtualSize, d);
        return res;
     }

    
    public FieldVector<T> copy() {
        return new SparseFieldVector<T>(this);
    }

    
    public T dotProduct(FieldVector<T> v) throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        T res = field.getZero();
        OpenIntToFieldHashMap<T>.Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            res = res.add(v.getEntry(iter.key()).multiply(iter.value()));
        }
        return res;
    }

    
    public FieldVector<T> ebeDivide(FieldVector<T> v)
        throws DimensionMismatchException, MathArithmeticException {
        checkVectorDimensions(v.getDimension());
        SparseFieldVector<T> res = new SparseFieldVector<T>(this);
        OpenIntToFieldHashMap<T>.Iterator iter = res.entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            res.setEntry(iter.key(), iter.value().divide(v.getEntry(iter.key())));
        }
        return res;
    }

    
    public FieldVector<T> ebeMultiply(FieldVector<T> v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        SparseFieldVector<T> res = new SparseFieldVector<T>(this);
        OpenIntToFieldHashMap<T>.Iterator iter = res.entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            res.setEntry(iter.key(), iter.value().multiply(v.getEntry(iter.key())));
        }
        return res;
    }

    
    @Deprecated
    public T[] getData() {
        return toArray();
    }

    
    public int getDimension() {
        return virtualSize;
    }

    
    public T getEntry(int index) throws OutOfRangeException {
        checkIndex(index);
        return entries.get(index);
   }

    
    public Field<T> getField() {
        return field;
    }

    
    public FieldVector<T> getSubVector(int index, int n)
        throws OutOfRangeException, NotPositiveException {
        if (n < 0) {
            throw new NotPositiveException(LocalizedFormats.NUMBER_OF_ELEMENTS_SHOULD_BE_POSITIVE, n);
        }
        checkIndex(index);
        checkIndex(index + n - 1);
        SparseFieldVector<T> res = new SparseFieldVector<T>(field,n);
        int end = index + n;
        OpenIntToFieldHashMap<T>.Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            if (key >= index && key < end) {
                res.setEntry(key - index, iter.value());
            }
        }
        return res;
    }

    
    public FieldVector<T> mapAdd(T d) throws NullArgumentException {
        return copy().mapAddToSelf(d);
    }

    
    public FieldVector<T> mapAddToSelf(T d) throws NullArgumentException {
        for (int i = 0; i < virtualSize; i++) {
            setEntry(i, getEntry(i).add(d));
        }
        return this;
    }

    
    public FieldVector<T> mapDivide(T d)
        throws NullArgumentException, MathArithmeticException {
        return copy().mapDivideToSelf(d);
    }

    
    public FieldVector<T> mapDivideToSelf(T d)
        throws NullArgumentException, MathArithmeticException {
        OpenIntToFieldHashMap<T>.Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            entries.put(iter.key(), iter.value().divide(d));
        }
        return this;
    }

    
    public FieldVector<T> mapInv() throws MathArithmeticException {
        return copy().mapInvToSelf();
    }

    
    public FieldVector<T> mapInvToSelf() throws MathArithmeticException {
        for (int i = 0; i < virtualSize; i++) {
            setEntry(i, field.getOne().divide(getEntry(i)));
        }
        return this;
    }

    
    public FieldVector<T> mapMultiply(T d) throws NullArgumentException {
        return copy().mapMultiplyToSelf(d);
    }

    
    public FieldVector<T> mapMultiplyToSelf(T d) throws NullArgumentException {
        OpenIntToFieldHashMap<T>.Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            entries.put(iter.key(), iter.value().multiply(d));
        }
        return this;
    }

    
    public FieldVector<T> mapSubtract(T d) throws NullArgumentException {
        return copy().mapSubtractToSelf(d);
    }

    
    public FieldVector<T> mapSubtractToSelf(T d) throws NullArgumentException {
        return mapAddToSelf(field.getZero().subtract(d));
    }

    
    public FieldMatrix<T> outerProduct(SparseFieldVector<T> v) {
        final int n = v.getDimension();
        SparseFieldMatrix<T> res = new SparseFieldMatrix<T>(field, virtualSize, n);
        OpenIntToFieldHashMap<T>.Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            OpenIntToFieldHashMap<T>.Iterator iter2 = v.entries.iterator();
            while (iter2.hasNext()) {
                iter2.advance();
                res.setEntry(iter.key(), iter2.key(), iter.value().multiply(iter2.value()));
            }
        }
        return res;
    }

    
    public FieldMatrix<T> outerProduct(FieldVector<T> v) {
        if (v instanceof SparseFieldVector<?>) {
            return outerProduct((SparseFieldVector<T>)v);
        } else {
            final int n = v.getDimension();
            FieldMatrix<T> res = new SparseFieldMatrix<T>(field, virtualSize, n);
            OpenIntToFieldHashMap<T>.Iterator iter = entries.iterator();
            while (iter.hasNext()) {
                iter.advance();
                int row = iter.key();
                FieldElement<T>value = iter.value();
                for (int col = 0; col < n; col++) {
                    res.setEntry(row, col, value.multiply(v.getEntry(col)));
                }
            }
            return res;
        }
    }

    
    public FieldVector<T> projection(FieldVector<T> v)
        throws DimensionMismatchException, MathArithmeticException {
        checkVectorDimensions(v.getDimension());
        return v.mapMultiply(dotProduct(v).divide(v.dotProduct(v)));
    }

    
    public void set(T value) {
        MathUtils.checkNotNull(value);
        for (int i = 0; i < virtualSize; i++) {
            setEntry(i, value);
        }
    }

    
    public void setEntry(int index, T value) throws NullArgumentException, OutOfRangeException {
        MathUtils.checkNotNull(value);
        checkIndex(index);
        entries.put(index, value);
    }

    
    public void setSubVector(int index, FieldVector<T> v)
        throws OutOfRangeException {
        checkIndex(index);
        checkIndex(index + v.getDimension() - 1);
        final int n = v.getDimension();
        for (int i = 0; i < n; i++) {
            setEntry(i + index, v.getEntry(i));
        }
    }

    
    public SparseFieldVector<T> subtract(SparseFieldVector<T> v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        SparseFieldVector<T> res = (SparseFieldVector<T>)copy();
        OpenIntToFieldHashMap<T>.Iterator iter = v.getEntries().iterator();
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            if (entries.containsKey(key)) {
                res.setEntry(key, entries.get(key).subtract(iter.value()));
            } else {
                res.setEntry(key, field.getZero().subtract(iter.value()));
            }
        }
        return res;
    }

    
    public FieldVector<T> subtract(FieldVector<T> v)
        throws DimensionMismatchException {
        if (v instanceof SparseFieldVector<?>) {
            return subtract((SparseFieldVector<T>)v);
        } else {
            final int n = v.getDimension();
            checkVectorDimensions(n);
            SparseFieldVector<T> res = new SparseFieldVector<T>(this);
            for (int i = 0; i < n; i++) {
                if (entries.containsKey(i)) {
                    res.setEntry(i, entries.get(i).subtract(v.getEntry(i)));
                } else {
                    res.setEntry(i, field.getZero().subtract(v.getEntry(i)));
                }
            }
            return res;
        }
    }

    
    public T[] toArray() {
        T[] res = MathArrays.buildArray(field, virtualSize);
        OpenIntToFieldHashMap<T>.Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            res[iter.key()] = iter.value();
        }
        return res;
    }

    
    private void checkIndex(final int index) throws OutOfRangeException {
        if (index < 0 || index >= getDimension()) {
            throw new OutOfRangeException(index, 0, getDimension() - 1);
        }
    }

    
    private void checkIndices(final int start, final int end)
        throws NumberIsTooSmallException, OutOfRangeException {
        final int dim = getDimension();
        if ((start < 0) || (start >= dim)) {
            throw new OutOfRangeException(LocalizedFormats.INDEX, start, 0,
                                          dim - 1);
        }
        if ((end < 0) || (end >= dim)) {
            throw new OutOfRangeException(LocalizedFormats.INDEX, end, 0,
                                          dim - 1);
        }
        if (end < start) {
            throw new NumberIsTooSmallException(LocalizedFormats.INITIAL_ROW_AFTER_FINAL_ROW,
                                                end, start, false);
        }
    }

    
    protected void checkVectorDimensions(int n)
        throws DimensionMismatchException {
        if (getDimension() != n) {
            throw new DimensionMismatchException(getDimension(), n);
        }
    }

    
    public FieldVector<T> add(FieldVector<T> v) throws DimensionMismatchException {
        if (v instanceof SparseFieldVector<?>) {
            return add((SparseFieldVector<T>) v);
        } else {
            final int n = v.getDimension();
            checkVectorDimensions(n);
            SparseFieldVector<T> res = new SparseFieldVector<T>(field,
                                                                getDimension());
            for (int i = 0; i < n; i++) {
                res.setEntry(i, v.getEntry(i).add(getEntry(i)));
            }
            return res;
        }
    }

    
    public T walkInDefaultOrder(final FieldVectorPreservingVisitor<T> visitor) {
        final int dim = getDimension();
        visitor.start(dim, 0, dim - 1);
        for (int i = 0; i < dim; i++) {
            visitor.visit(i, getEntry(i));
        }
        return visitor.end();
    }

    
    public T walkInDefaultOrder(final FieldVectorPreservingVisitor<T> visitor,
                                final int start, final int end)
        throws NumberIsTooSmallException, OutOfRangeException {
        checkIndices(start, end);
        visitor.start(getDimension(), start, end);
        for (int i = start; i <= end; i++) {
            visitor.visit(i, getEntry(i));
        }
        return visitor.end();
    }

    
    public T walkInOptimizedOrder(final FieldVectorPreservingVisitor<T> visitor) {
        return walkInDefaultOrder(visitor);
    }

    
    public T walkInOptimizedOrder(final FieldVectorPreservingVisitor<T> visitor,
                                  final int start, final int end)
        throws NumberIsTooSmallException, OutOfRangeException {
        return walkInDefaultOrder(visitor, start, end);
    }

    
    public T walkInDefaultOrder(final FieldVectorChangingVisitor<T> visitor) {
        final int dim = getDimension();
        visitor.start(dim, 0, dim - 1);
        for (int i = 0; i < dim; i++) {
            setEntry(i, visitor.visit(i, getEntry(i)));
        }
        return visitor.end();
    }

    
    public T walkInDefaultOrder(final FieldVectorChangingVisitor<T> visitor,
                                final int start, final int end)
        throws NumberIsTooSmallException, OutOfRangeException {
        checkIndices(start, end);
        visitor.start(getDimension(), start, end);
        for (int i = start; i <= end; i++) {
            setEntry(i, visitor.visit(i, getEntry(i)));
        }
        return visitor.end();
    }

    
    public T walkInOptimizedOrder(final FieldVectorChangingVisitor<T> visitor) {
        return walkInDefaultOrder(visitor);
    }

    
    public T walkInOptimizedOrder(final FieldVectorChangingVisitor<T> visitor,
                                  final int start, final int end)
        throws NumberIsTooSmallException, OutOfRangeException {
        return walkInDefaultOrder(visitor, start, end);
    }

    
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((field == null) ? 0 : field.hashCode());
        result = prime * result + virtualSize;
        OpenIntToFieldHashMap<T>.Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            int temp = iter.value().hashCode();
            result = prime * result + temp;
        }
        return result;
    }


    
    @Override
    public boolean equals(Object obj) {

        if (this == obj) {
            return true;
        }

        if (!(obj instanceof SparseFieldVector<?>)) {
            return false;
        }

        @SuppressWarnings("unchecked") // OK, because "else if" check below ensures that
                                       // other must be the same type as this
        SparseFieldVector<T> other = (SparseFieldVector<T>) obj;
        if (field == null) {
            if (other.field != null) {
                return false;
            }
        } else if (!field.equals(other.field)) {
            return false;
        }
        if (virtualSize != other.virtualSize) {
            return false;
        }

        OpenIntToFieldHashMap<T>.Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            T test = other.getEntry(iter.key());
            if (!test.equals(iter.value())) {
                return false;
            }
        }
        iter = other.getEntries().iterator();
        while (iter.hasNext()) {
            iter.advance();
            T test = iter.value();
            if (!test.equals(getEntry(iter.key()))) {
                return false;
            }
        }
        return true;
    }
}
