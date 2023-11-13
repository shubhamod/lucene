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

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class ArrayFieldVector<T extends FieldElement<T>> implements FieldVector<T>, Serializable {
    
    private static final long serialVersionUID = 7648186910365927050L;

    
    private T[] data;

    
    private final Field<T> field;

    
    public ArrayFieldVector(final Field<T> field) {
        this(field, 0);
    }

    
    public ArrayFieldVector(Field<T> field, int size) {
        this.field = field;
        this.data  = MathArrays.buildArray(field, size);
    }

    
    public ArrayFieldVector(int size, T preset) {
        this(preset.getField(), size);
        Arrays.fill(data, preset);
    }

    
    public ArrayFieldVector(T[] d)
            throws NullArgumentException, ZeroException {
        MathUtils.checkNotNull(d);
        try {
            field = d[0].getField();
            data = d.clone();
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ZeroException(LocalizedFormats.VECTOR_MUST_HAVE_AT_LEAST_ONE_ELEMENT);
        }
    }

    
    public ArrayFieldVector(Field<T> field, T[] d)
            throws NullArgumentException {
        MathUtils.checkNotNull(d);
        this.field = field;
        data = d.clone();
    }

    
    public ArrayFieldVector(T[] d, boolean copyArray)
            throws NullArgumentException, ZeroException {
        MathUtils.checkNotNull(d);
        if (d.length == 0) {
            throw new ZeroException(LocalizedFormats.VECTOR_MUST_HAVE_AT_LEAST_ONE_ELEMENT);
        }
        field = d[0].getField();
        data = copyArray ? d.clone() : d;
    }

    
    public ArrayFieldVector(Field<T> field, T[] d, boolean copyArray)
            throws NullArgumentException {
        MathUtils.checkNotNull(d);
        this.field = field;
        data = copyArray ? d.clone() :  d;
    }

    
    public ArrayFieldVector(T[] d, int pos, int size)
            throws NullArgumentException, NumberIsTooLargeException {
        MathUtils.checkNotNull(d);
        if (d.length < pos + size) {
            throw new NumberIsTooLargeException(pos + size, d.length, true);
        }
        field = d[0].getField();
        data = MathArrays.buildArray(field, size);
        System.arraycopy(d, pos, data, 0, size);
    }

    
    public ArrayFieldVector(Field<T> field, T[] d, int pos, int size)
            throws NullArgumentException, NumberIsTooLargeException {
        MathUtils.checkNotNull(d);
        if (d.length < pos + size) {
            throw new NumberIsTooLargeException(pos + size, d.length, true);
        }
        this.field = field;
        data = MathArrays.buildArray(field, size);
        System.arraycopy(d, pos, data, 0, size);
    }

    
    public ArrayFieldVector(FieldVector<T> v)
            throws NullArgumentException {
        MathUtils.checkNotNull(v);
        field = v.getField();
        data = MathArrays.buildArray(field, v.getDimension());
        for (int i = 0; i < data.length; ++i) {
            data[i] = v.getEntry(i);
        }
    }

    
    public ArrayFieldVector(ArrayFieldVector<T> v)
            throws NullArgumentException {
        MathUtils.checkNotNull(v);
        field = v.getField();
        data = v.data.clone();
    }

    
    public ArrayFieldVector(ArrayFieldVector<T> v, boolean deep)
            throws NullArgumentException {
        MathUtils.checkNotNull(v);
        field = v.getField();
        data = deep ? v.data.clone() : v.data;
    }

    
    @Deprecated
    public ArrayFieldVector(ArrayFieldVector<T> v1, ArrayFieldVector<T> v2)
            throws NullArgumentException {
        this((FieldVector<T>) v1, (FieldVector<T>) v2);
    }

    
    public ArrayFieldVector(FieldVector<T> v1, FieldVector<T> v2)
            throws NullArgumentException {
        MathUtils.checkNotNull(v1);
        MathUtils.checkNotNull(v2);
        field = v1.getField();
        final T[] v1Data =
                (v1 instanceof ArrayFieldVector) ? ((ArrayFieldVector<T>) v1).data : v1.toArray();
        final T[] v2Data =
                (v2 instanceof ArrayFieldVector) ? ((ArrayFieldVector<T>) v2).data : v2.toArray();
        data = MathArrays.buildArray(field, v1Data.length + v2Data.length);
        System.arraycopy(v1Data, 0, data, 0, v1Data.length);
        System.arraycopy(v2Data, 0, data, v1Data.length, v2Data.length);
    }

    
    @Deprecated
    public ArrayFieldVector(ArrayFieldVector<T> v1, T[] v2)
            throws NullArgumentException {
        this((FieldVector<T>) v1, v2);
    }

    
    public ArrayFieldVector(FieldVector<T> v1, T[] v2)
            throws NullArgumentException {
        MathUtils.checkNotNull(v1);
        MathUtils.checkNotNull(v2);
        field = v1.getField();
        final T[] v1Data =
                (v1 instanceof ArrayFieldVector) ? ((ArrayFieldVector<T>) v1).data : v1.toArray();
        data = MathArrays.buildArray(field, v1Data.length + v2.length);
        System.arraycopy(v1Data, 0, data, 0, v1Data.length);
        System.arraycopy(v2, 0, data, v1Data.length, v2.length);
    }

    
    @Deprecated
    public ArrayFieldVector(T[] v1, ArrayFieldVector<T> v2)
            throws NullArgumentException {
        this(v1, (FieldVector<T>) v2);
    }

    
    public ArrayFieldVector(T[] v1, FieldVector<T> v2)
            throws NullArgumentException {
        MathUtils.checkNotNull(v1);
        MathUtils.checkNotNull(v2);
        field = v2.getField();
        final T[] v2Data =
                (v2 instanceof ArrayFieldVector) ? ((ArrayFieldVector<T>) v2).data : v2.toArray();
        data = MathArrays.buildArray(field, v1.length + v2Data.length);
        System.arraycopy(v1, 0, data, 0, v1.length);
        System.arraycopy(v2Data, 0, data, v1.length, v2Data.length);
    }

    
    public ArrayFieldVector(T[] v1, T[] v2)
            throws NullArgumentException, ZeroException {
        MathUtils.checkNotNull(v1);
        MathUtils.checkNotNull(v2);
        if (v1.length + v2.length == 0) {
            throw new ZeroException(LocalizedFormats.VECTOR_MUST_HAVE_AT_LEAST_ONE_ELEMENT);
        }
        data = MathArrays.buildArray(v1[0].getField(), v1.length + v2.length);
        System.arraycopy(v1, 0, data, 0, v1.length);
        System.arraycopy(v2, 0, data, v1.length, v2.length);
        field = data[0].getField();
    }

    
    public ArrayFieldVector(Field<T> field, T[] v1, T[] v2)
            throws NullArgumentException, ZeroException {
        MathUtils.checkNotNull(v1);
        MathUtils.checkNotNull(v2);
        if (v1.length + v2.length == 0) {
            throw new ZeroException(LocalizedFormats.VECTOR_MUST_HAVE_AT_LEAST_ONE_ELEMENT);
        }
        data = MathArrays.buildArray(field, v1.length + v2.length);
        System.arraycopy(v1, 0, data, 0, v1.length);
        System.arraycopy(v2, 0, data, v1.length, v2.length);
        this.field = field;
    }

    
    public Field<T> getField() {
        return field;
    }

    
    public FieldVector<T> copy() {
        return new ArrayFieldVector<T>(this, true);
    }

    
    public FieldVector<T> add(FieldVector<T> v)
        throws DimensionMismatchException {
        try {
            return add((ArrayFieldVector<T>) v);
        } catch (ClassCastException cce) {
            checkVectorDimensions(v);
            T[] out = MathArrays.buildArray(field, data.length);
            for (int i = 0; i < data.length; i++) {
                out[i] = data[i].add(v.getEntry(i));
            }
            return new ArrayFieldVector<T>(field, out, false);
        }
    }

    
    public ArrayFieldVector<T> add(ArrayFieldVector<T> v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.data.length);
        T[] out = MathArrays.buildArray(field, data.length);
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i].add(v.data[i]);
        }
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public FieldVector<T> subtract(FieldVector<T> v)
        throws DimensionMismatchException {
        try {
            return subtract((ArrayFieldVector<T>) v);
        } catch (ClassCastException cce) {
            checkVectorDimensions(v);
            T[] out = MathArrays.buildArray(field, data.length);
            for (int i = 0; i < data.length; i++) {
                out[i] = data[i].subtract(v.getEntry(i));
            }
            return new ArrayFieldVector<T>(field, out, false);
        }
    }

    
    public ArrayFieldVector<T> subtract(ArrayFieldVector<T> v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.data.length);
        T[] out = MathArrays.buildArray(field, data.length);
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i].subtract(v.data[i]);
        }
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public FieldVector<T> mapAdd(T d) throws NullArgumentException {
        T[] out = MathArrays.buildArray(field, data.length);
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i].add(d);
        }
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public FieldVector<T> mapAddToSelf(T d) throws NullArgumentException {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i].add(d);
        }
        return this;
    }

    
    public FieldVector<T> mapSubtract(T d) throws NullArgumentException {
        T[] out = MathArrays.buildArray(field, data.length);
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i].subtract(d);
        }
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public FieldVector<T> mapSubtractToSelf(T d) throws NullArgumentException {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i].subtract(d);
        }
        return this;
    }

    
    public FieldVector<T> mapMultiply(T d) throws NullArgumentException {
        T[] out = MathArrays.buildArray(field, data.length);
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i].multiply(d);
        }
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public FieldVector<T> mapMultiplyToSelf(T d) throws NullArgumentException {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i].multiply(d);
        }
        return this;
    }

    
    public FieldVector<T> mapDivide(T d)
        throws NullArgumentException, MathArithmeticException {
        MathUtils.checkNotNull(d);
        T[] out = MathArrays.buildArray(field, data.length);
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i].divide(d);
        }
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public FieldVector<T> mapDivideToSelf(T d)
        throws NullArgumentException, MathArithmeticException {
        MathUtils.checkNotNull(d);
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i].divide(d);
        }
        return this;
    }

    
    public FieldVector<T> mapInv() throws MathArithmeticException {
        T[] out = MathArrays.buildArray(field, data.length);
        final T one = field.getOne();
        for (int i = 0; i < data.length; i++) {
            try {
                out[i] = one.divide(data[i]);
            } catch (final MathArithmeticException e) {
                throw new MathArithmeticException(LocalizedFormats.INDEX, i);
            }
        }
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public FieldVector<T> mapInvToSelf() throws MathArithmeticException {
        final T one = field.getOne();
        for (int i = 0; i < data.length; i++) {
            try {
                data[i] = one.divide(data[i]);
            } catch (final MathArithmeticException e) {
                throw new MathArithmeticException(LocalizedFormats.INDEX, i);
            }
        }
        return this;
    }

    
    public FieldVector<T> ebeMultiply(FieldVector<T> v)
        throws DimensionMismatchException {
        try {
            return ebeMultiply((ArrayFieldVector<T>) v);
        } catch (ClassCastException cce) {
            checkVectorDimensions(v);
            T[] out = MathArrays.buildArray(field, data.length);
            for (int i = 0; i < data.length; i++) {
                out[i] = data[i].multiply(v.getEntry(i));
            }
            return new ArrayFieldVector<T>(field, out, false);
        }
    }

    
    public ArrayFieldVector<T> ebeMultiply(ArrayFieldVector<T> v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.data.length);
        T[] out = MathArrays.buildArray(field, data.length);
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i].multiply(v.data[i]);
        }
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public FieldVector<T> ebeDivide(FieldVector<T> v)
        throws DimensionMismatchException, MathArithmeticException {
        try {
            return ebeDivide((ArrayFieldVector<T>) v);
        } catch (ClassCastException cce) {
            checkVectorDimensions(v);
            T[] out = MathArrays.buildArray(field, data.length);
            for (int i = 0; i < data.length; i++) {
                try {
                    out[i] = data[i].divide(v.getEntry(i));
                } catch (final MathArithmeticException e) {
                    throw new MathArithmeticException(LocalizedFormats.INDEX, i);
                }
            }
            return new ArrayFieldVector<T>(field, out, false);
        }
    }

    
    public ArrayFieldVector<T> ebeDivide(ArrayFieldVector<T> v)
        throws DimensionMismatchException, MathArithmeticException {
        checkVectorDimensions(v.data.length);
        T[] out = MathArrays.buildArray(field, data.length);
        for (int i = 0; i < data.length; i++) {
            try {
                out[i] = data[i].divide(v.data[i]);
            } catch (final MathArithmeticException e) {
                throw new MathArithmeticException(LocalizedFormats.INDEX, i);
            }
        }
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public T[] getData() {
        return data.clone();
    }

    
    public T[] getDataRef() {
        return data;
    }

    
    public T dotProduct(FieldVector<T> v)
        throws DimensionMismatchException {
        try {
            return dotProduct((ArrayFieldVector<T>) v);
        } catch (ClassCastException cce) {
            checkVectorDimensions(v);
            T dot = field.getZero();
            for (int i = 0; i < data.length; i++) {
                dot = dot.add(data[i].multiply(v.getEntry(i)));
            }
            return dot;
        }
    }

    
    public T dotProduct(ArrayFieldVector<T> v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.data.length);
        T dot = field.getZero();
        for (int i = 0; i < data.length; i++) {
            dot = dot.add(data[i].multiply(v.data[i]));
        }
        return dot;
    }

    
    public FieldVector<T> projection(FieldVector<T> v)
        throws DimensionMismatchException, MathArithmeticException {
        return v.mapMultiply(dotProduct(v).divide(v.dotProduct(v)));
    }

    
    public ArrayFieldVector<T> projection(ArrayFieldVector<T> v)
        throws DimensionMismatchException, MathArithmeticException {
        return (ArrayFieldVector<T>) v.mapMultiply(dotProduct(v).divide(v.dotProduct(v)));
    }

    
    public FieldMatrix<T> outerProduct(FieldVector<T> v) {
        try {
            return outerProduct((ArrayFieldVector<T>) v);
        } catch (ClassCastException cce) {
            final int m = data.length;
            final int n = v.getDimension();
            final FieldMatrix<T> out = new Array2DRowFieldMatrix<T>(field, m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    out.setEntry(i, j, data[i].multiply(v.getEntry(j)));
                }
            }
            return out;
        }
    }

    
    public FieldMatrix<T> outerProduct(ArrayFieldVector<T> v) {
        final int m = data.length;
        final int n = v.data.length;
        final FieldMatrix<T> out = new Array2DRowFieldMatrix<T>(field, m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                out.setEntry(i, j, data[i].multiply(v.data[j]));
            }
        }
        return out;
    }

    
    public T getEntry(int index) {
        return data[index];
    }

    
    public int getDimension() {
        return data.length;
    }

    
    public FieldVector<T> append(FieldVector<T> v) {
        try {
            return append((ArrayFieldVector<T>) v);
        } catch (ClassCastException cce) {
            return new ArrayFieldVector<T>(this,new ArrayFieldVector<T>(v));
        }
    }

    
    public ArrayFieldVector<T> append(ArrayFieldVector<T> v) {
        return new ArrayFieldVector<T>(this, v);
    }

    
    public FieldVector<T> append(T in) {
        final T[] out = MathArrays.buildArray(field, data.length + 1);
        System.arraycopy(data, 0, out, 0, data.length);
        out[data.length] = in;
        return new ArrayFieldVector<T>(field, out, false);
    }

    
    public FieldVector<T> getSubVector(int index, int n)
        throws OutOfRangeException, NotPositiveException {
        if (n < 0) {
            throw new NotPositiveException(LocalizedFormats.NUMBER_OF_ELEMENTS_SHOULD_BE_POSITIVE, n);
        }
        ArrayFieldVector<T> out = new ArrayFieldVector<T>(field, n);
        try {
            System.arraycopy(data, index, out.data, 0, n);
        } catch (IndexOutOfBoundsException e) {
            checkIndex(index);
            checkIndex(index + n - 1);
        }
        return out;
    }

    
    public void setEntry(int index, T value) {
        try {
            data[index] = value;
        } catch (IndexOutOfBoundsException e) {
            checkIndex(index);
        }
    }

    
    public void setSubVector(int index, FieldVector<T> v) throws OutOfRangeException {
        try {
            try {
                set(index, (ArrayFieldVector<T>) v);
            } catch (ClassCastException cce) {
                for (int i = index; i < index + v.getDimension(); ++i) {
                    data[i] = v.getEntry(i-index);
                }
            }
        } catch (IndexOutOfBoundsException e) {
            checkIndex(index);
            checkIndex(index + v.getDimension() - 1);
        }
    }

    
    public void set(int index, ArrayFieldVector<T> v) throws OutOfRangeException {
        try {
            System.arraycopy(v.data, 0, data, index, v.data.length);
        } catch (IndexOutOfBoundsException e) {
            checkIndex(index);
            checkIndex(index + v.data.length - 1);
        }
    }

    
    public void set(T value) {
        Arrays.fill(data, value);
    }

    
    public T[] toArray(){
        return data.clone();
    }

    
    protected void checkVectorDimensions(FieldVector<T> v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
    }

    
    protected void checkVectorDimensions(int n)
        throws DimensionMismatchException {
        if (data.length != n) {
            throw new DimensionMismatchException(data.length, n);
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
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other == null) {
            return false;
        }

        try {
            @SuppressWarnings("unchecked") // May fail, but we ignore ClassCastException
                FieldVector<T> rhs = (FieldVector<T>) other;
            if (data.length != rhs.getDimension()) {
                return false;
            }

            for (int i = 0; i < data.length; ++i) {
                if (!data[i].equals(rhs.getEntry(i))) {
                    return false;
                }
            }
            return true;
        } catch (ClassCastException ex) {
            // ignore exception
            return false;
        }
    }

    
    @Override
    public int hashCode() {
        int h = 3542;
        for (final T a : data) {
            h ^= a.hashCode();
        }
        return h;
    }

    
    private void checkIndex(final int index) throws OutOfRangeException {
        if (index < 0 || index >= getDimension()) {
            throw new OutOfRangeException(LocalizedFormats.INDEX,
                                          index, 0, getDimension() - 1);
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

}
