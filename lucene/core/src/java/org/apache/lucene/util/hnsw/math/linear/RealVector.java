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

import java.util.Iterator;
import java.util.NoSuchElementException;

import org.apache.lucene.util.hnsw.math.exception.MathUnsupportedOperationException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.analysis.FunctionUtils;
import org.apache.lucene.util.hnsw.math.analysis.function.Add;
import org.apache.lucene.util.hnsw.math.analysis.function.Multiply;
import org.apache.lucene.util.hnsw.math.analysis.function.Divide;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public abstract class RealVector {
    
    public abstract int getDimension();

    
    public abstract double getEntry(int index) throws OutOfRangeException;

    
    public abstract void setEntry(int index, double value)
        throws OutOfRangeException;

    
    public void addToEntry(int index, double increment)
        throws OutOfRangeException {
        setEntry(index, getEntry(index) + increment);
    }

    
    public abstract RealVector append(RealVector v);

    
    public abstract RealVector append(double d);

    
    public abstract RealVector getSubVector(int index, int n)
        throws NotPositiveException, OutOfRangeException;

    
    public abstract void setSubVector(int index, RealVector v)
        throws OutOfRangeException;

    
    public abstract boolean isNaN();

    
    public abstract boolean isInfinite();

    
    protected void checkVectorDimensions(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
    }

    
    protected void checkVectorDimensions(int n)
        throws DimensionMismatchException {
        int d = getDimension();
        if (d != n) {
            throw new DimensionMismatchException(d, n);
        }
    }

    
    protected void checkIndex(final int index) throws OutOfRangeException {
        if (index < 0 ||
            index >= getDimension()) {
            throw new OutOfRangeException(LocalizedFormats.INDEX,
                                          index, 0, getDimension() - 1);
        }
    }

    
    protected void checkIndices(final int start, final int end)
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
            // TODO Use more specific error message
            throw new NumberIsTooSmallException(LocalizedFormats.INITIAL_ROW_AFTER_FINAL_ROW,
                                                end, start, false);
        }
    }

    
    public RealVector add(RealVector v) throws DimensionMismatchException {
        checkVectorDimensions(v);
        RealVector result = v.copy();
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            final int index = e.getIndex();
            result.setEntry(index, e.getValue() + result.getEntry(index));
        }
        return result;
    }

    
    public RealVector subtract(RealVector v) throws DimensionMismatchException {
        checkVectorDimensions(v);
        RealVector result = v.mapMultiply(-1d);
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            final int index = e.getIndex();
            result.setEntry(index, e.getValue() + result.getEntry(index));
        }
        return result;
    }

    
    public RealVector mapAdd(double d) {
        return copy().mapAddToSelf(d);
    }

    
    public RealVector mapAddToSelf(double d) {
        if (d != 0) {
            return mapToSelf(FunctionUtils.fix2ndArgument(new Add(), d));
        }
        return this;
    }

    
    public abstract RealVector copy();

    
    public double dotProduct(RealVector v) throws DimensionMismatchException {
        checkVectorDimensions(v);
        double d = 0;
        final int n = getDimension();
        for (int i = 0; i < n; i++) {
            d += getEntry(i) * v.getEntry(i);
        }
        return d;
    }

    
    public double cosine(RealVector v) throws DimensionMismatchException,
        MathArithmeticException {
        final double norm = getNorm();
        final double vNorm = v.getNorm();

        if (norm == 0 ||
            vNorm == 0) {
            throw new MathArithmeticException(LocalizedFormats.ZERO_NORM);
        }
        return dotProduct(v) / (norm * vNorm);
    }

    
    public abstract RealVector ebeDivide(RealVector v)
        throws DimensionMismatchException;

    
    public abstract RealVector ebeMultiply(RealVector v)
        throws DimensionMismatchException;

    
    public double getDistance(RealVector v) throws DimensionMismatchException {
        checkVectorDimensions(v);
        double d = 0;
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            final double diff = e.getValue() - v.getEntry(e.getIndex());
            d += diff * diff;
        }
        return FastMath.sqrt(d);
    }

    
    public double getNorm() {
        double sum = 0;
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            final double value = e.getValue();
            sum += value * value;
        }
        return FastMath.sqrt(sum);
    }

    
    public double getL1Norm() {
        double norm = 0;
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            norm += FastMath.abs(e.getValue());
        }
        return norm;
    }

    
    public double getLInfNorm() {
        double norm = 0;
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            norm = FastMath.max(norm, FastMath.abs(e.getValue()));
        }
        return norm;
    }

    
    public double getL1Distance(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v);
        double d = 0;
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            d += FastMath.abs(e.getValue() - v.getEntry(e.getIndex()));
        }
        return d;
    }

    
    public double getLInfDistance(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v);
        double d = 0;
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            d = FastMath.max(FastMath.abs(e.getValue() - v.getEntry(e.getIndex())), d);
        }
        return d;
    }

    
    public int getMinIndex() {
        int minIndex    = -1;
        double minValue = Double.POSITIVE_INFINITY;
        Iterator<Entry> iterator = iterator();
        while (iterator.hasNext()) {
            final Entry entry = iterator.next();
            if (entry.getValue() <= minValue) {
                minIndex = entry.getIndex();
                minValue = entry.getValue();
            }
        }
        return minIndex;
    }

    
    public double getMinValue() {
        final int minIndex = getMinIndex();
        return minIndex < 0 ? Double.NaN : getEntry(minIndex);
    }

    
    public int getMaxIndex() {
        int maxIndex    = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        Iterator<Entry> iterator = iterator();
        while (iterator.hasNext()) {
            final Entry entry = iterator.next();
            if (entry.getValue() >= maxValue) {
                maxIndex = entry.getIndex();
                maxValue = entry.getValue();
            }
        }
        return maxIndex;
    }

    
    public double getMaxValue() {
        final int maxIndex = getMaxIndex();
        return maxIndex < 0 ? Double.NaN : getEntry(maxIndex);
    }


    
    public RealVector mapMultiply(double d) {
        return copy().mapMultiplyToSelf(d);
    }

    
    public RealVector mapMultiplyToSelf(double d){
        return mapToSelf(FunctionUtils.fix2ndArgument(new Multiply(), d));
    }

    
    public RealVector mapSubtract(double d) {
        return copy().mapSubtractToSelf(d);
    }

    
    public RealVector mapSubtractToSelf(double d){
        return mapAddToSelf(-d);
    }

    
    public RealVector mapDivide(double d) {
        return copy().mapDivideToSelf(d);
    }

    
    public RealVector mapDivideToSelf(double d){
        return mapToSelf(FunctionUtils.fix2ndArgument(new Divide(), d));
    }

    
    public RealMatrix outerProduct(RealVector v) {
        final int m = this.getDimension();
        final int n = v.getDimension();
        final RealMatrix product;
        if (v instanceof SparseRealVector || this instanceof SparseRealVector) {
            product = new OpenMapRealMatrix(m, n);
        } else {
            product = new Array2DRowRealMatrix(m, n);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                product.setEntry(i, j, this.getEntry(i) * v.getEntry(j));
            }
        }
        return product;
    }

    
    public RealVector projection(final RealVector v)
        throws DimensionMismatchException, MathArithmeticException {
        final double norm2 = v.dotProduct(v);
        if (norm2 == 0.0) {
            throw new MathArithmeticException(LocalizedFormats.ZERO_NORM);
        }
        return v.mapMultiply(dotProduct(v) / v.dotProduct(v));
    }

    
    public void set(double value) {
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            e.setValue(value);
        }
    }

    
    public double[] toArray() {
        int dim = getDimension();
        double[] values = new double[dim];
        for (int i = 0; i < dim; i++) {
            values[i] = getEntry(i);
        }
        return values;
    }

    
    public RealVector unitVector() throws MathArithmeticException {
        final double norm = getNorm();
        if (norm == 0) {
            throw new MathArithmeticException(LocalizedFormats.ZERO_NORM);
        }
        return mapDivide(norm);
    }

    
    public void unitize() throws MathArithmeticException {
        final double norm = getNorm();
        if (norm == 0) {
            throw new MathArithmeticException(LocalizedFormats.ZERO_NORM);
        }
        mapDivideToSelf(getNorm());
    }

    
    public Iterator<Entry> sparseIterator() {
        return new SparseEntryIterator();
    }

    
    public Iterator<Entry> iterator() {
        final int dim = getDimension();
        return new Iterator<Entry>() {

            
            private int i = 0;

            
            private Entry e = new Entry();

            
            public boolean hasNext() {
                return i < dim;
            }

            
            public Entry next() {
                if (i < dim) {
                    e.setIndex(i++);
                    return e;
                } else {
                    throw new NoSuchElementException();
                }
            }

            
            public void remove() throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }
        };
    }

    
    public RealVector map(UnivariateFunction function) {
        return copy().mapToSelf(function);
    }

    
    public RealVector mapToSelf(UnivariateFunction function) {
        Iterator<Entry> it = iterator();
        while (it.hasNext()) {
            final Entry e = it.next();
            e.setValue(function.value(e.getValue()));
        }
        return this;
    }

    
    public RealVector combine(double a, double b, RealVector y)
        throws DimensionMismatchException {
        return copy().combineToSelf(a, b, y);
    }

    
    public RealVector combineToSelf(double a, double b, RealVector y)
        throws DimensionMismatchException {
        checkVectorDimensions(y);
        for (int i = 0; i < getDimension(); i++) {
            final double xi = getEntry(i);
            final double yi = y.getEntry(i);
            setEntry(i, a * xi + b * yi);
        }
        return this;
    }

    
    public double walkInDefaultOrder(final RealVectorPreservingVisitor visitor) {
        final int dim = getDimension();
        visitor.start(dim, 0, dim - 1);
        for (int i = 0; i < dim; i++) {
            visitor.visit(i, getEntry(i));
        }
        return visitor.end();
    }

    
    public double walkInDefaultOrder(final RealVectorPreservingVisitor visitor,
                                     final int start, final int end)
        throws NumberIsTooSmallException, OutOfRangeException {
        checkIndices(start, end);
        visitor.start(getDimension(), start, end);
        for (int i = start; i <= end; i++) {
            visitor.visit(i, getEntry(i));
        }
        return visitor.end();
    }

    
    public double walkInOptimizedOrder(final RealVectorPreservingVisitor visitor) {
        return walkInDefaultOrder(visitor);
    }

    
    public double walkInOptimizedOrder(final RealVectorPreservingVisitor visitor,
                                       final int start, final int end)
        throws NumberIsTooSmallException, OutOfRangeException {
        return walkInDefaultOrder(visitor, start, end);
    }

    
    public double walkInDefaultOrder(final RealVectorChangingVisitor visitor) {
        final int dim = getDimension();
        visitor.start(dim, 0, dim - 1);
        for (int i = 0; i < dim; i++) {
            setEntry(i, visitor.visit(i, getEntry(i)));
        }
        return visitor.end();
    }

    
    public double walkInDefaultOrder(final RealVectorChangingVisitor visitor,
                              final int start, final int end)
        throws NumberIsTooSmallException, OutOfRangeException {
        checkIndices(start, end);
        visitor.start(getDimension(), start, end);
        for (int i = start; i <= end; i++) {
            setEntry(i, visitor.visit(i, getEntry(i)));
        }
        return visitor.end();
    }

    
    public double walkInOptimizedOrder(final RealVectorChangingVisitor visitor) {
        return walkInDefaultOrder(visitor);
    }

    
    public double walkInOptimizedOrder(final RealVectorChangingVisitor visitor,
                                       final int start, final int end)
        throws NumberIsTooSmallException, OutOfRangeException {
        return walkInDefaultOrder(visitor, start, end);
    }

    
    protected class Entry {
        
        private int index;

        
        public Entry() {
            setIndex(0);
        }

        
        public double getValue() {
            return getEntry(getIndex());
        }

        
        public void setValue(double value) {
            setEntry(getIndex(), value);
        }

        
        public int getIndex() {
            return index;
        }

        
        public void setIndex(int index) {
            this.index = index;
        }
    }

    
    @Override
    public boolean equals(Object other)
        throws MathUnsupportedOperationException {
        throw new MathUnsupportedOperationException();
    }

    
    @Override
    public int hashCode() throws MathUnsupportedOperationException {
        throw new MathUnsupportedOperationException();
    }

    
    protected class SparseEntryIterator implements Iterator<Entry> {
        
        private final int dim;
        
        private Entry current;
        
        private Entry next;

        
        protected SparseEntryIterator() {
            dim = getDimension();
            current = new Entry();
            next = new Entry();
            if (next.getValue() == 0) {
                advance(next);
            }
        }

        
        protected void advance(Entry e) {
            if (e == null) {
                return;
            }
            do {
                e.setIndex(e.getIndex() + 1);
            } while (e.getIndex() < dim && e.getValue() == 0);
            if (e.getIndex() >= dim) {
                e.setIndex(-1);
            }
        }

        
        public boolean hasNext() {
            return next.getIndex() >= 0;
        }

        
        public Entry next() {
            int index = next.getIndex();
            if (index < 0) {
                throw new NoSuchElementException();
            }
            current.setIndex(index);
            advance(next);
            return current;
        }

        
        public void remove() throws MathUnsupportedOperationException {
            throw new MathUnsupportedOperationException();
        }
    }

    
    public static RealVector unmodifiableRealVector(final RealVector v) {
        
        return new RealVector() {
            
            @Override
            public RealVector mapToSelf(UnivariateFunction function)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public RealVector map(UnivariateFunction function) {
                return v.map(function);
            }

            
            @Override
            public Iterator<Entry> iterator() {
                final Iterator<Entry> i = v.iterator();
                return new Iterator<Entry>() {
                    
                    private final UnmodifiableEntry e = new UnmodifiableEntry();

                    
                    public boolean hasNext() {
                        return i.hasNext();
                    }

                    
                    public Entry next() {
                        e.setIndex(i.next().getIndex());
                        return e;
                    }

                    
                    public void remove() throws MathUnsupportedOperationException {
                        throw new MathUnsupportedOperationException();
                    }
                };
            }

            
            @Override
            public Iterator<Entry> sparseIterator() {
                final Iterator<Entry> i = v.sparseIterator();

                return new Iterator<Entry>() {
                    
                    private final UnmodifiableEntry e = new UnmodifiableEntry();

                    
                    public boolean hasNext() {
                        return i.hasNext();
                    }

                    
                    public Entry next() {
                        e.setIndex(i.next().getIndex());
                        return e;
                    }

                    
                    public void remove()
                        throws MathUnsupportedOperationException {
                        throw new MathUnsupportedOperationException();
                    }
                };
            }

            
            @Override
            public RealVector copy() {
                return v.copy();
            }

            
            @Override
            public RealVector add(RealVector w)
                throws DimensionMismatchException {
                return v.add(w);
            }

            
            @Override
            public RealVector subtract(RealVector w)
                throws DimensionMismatchException {
                return v.subtract(w);
            }

            
            @Override
            public RealVector mapAdd(double d) {
                return v.mapAdd(d);
            }

            
            @Override
            public RealVector mapAddToSelf(double d)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public RealVector mapSubtract(double d) {
                return v.mapSubtract(d);
            }

            
            @Override
            public RealVector mapSubtractToSelf(double d)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public RealVector mapMultiply(double d) {
                return v.mapMultiply(d);
            }

            
            @Override
            public RealVector mapMultiplyToSelf(double d)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public RealVector mapDivide(double d) {
                return v.mapDivide(d);
            }

            
            @Override
            public RealVector mapDivideToSelf(double d)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public RealVector ebeMultiply(RealVector w)
                throws DimensionMismatchException {
                return v.ebeMultiply(w);
            }

            
            @Override
            public RealVector ebeDivide(RealVector w)
                throws DimensionMismatchException {
                return v.ebeDivide(w);
            }

            
            @Override
            public double dotProduct(RealVector w)
                throws DimensionMismatchException {
                return v.dotProduct(w);
            }

            
            @Override
            public double cosine(RealVector w)
                throws DimensionMismatchException, MathArithmeticException {
                return v.cosine(w);
            }

            
            @Override
            public double getNorm() {
                return v.getNorm();
            }

            
            @Override
            public double getL1Norm() {
                return v.getL1Norm();
            }

            
            @Override
            public double getLInfNorm() {
                return v.getLInfNorm();
            }

            
            @Override
            public double getDistance(RealVector w)
                throws DimensionMismatchException {
                return v.getDistance(w);
            }

            
            @Override
            public double getL1Distance(RealVector w)
                throws DimensionMismatchException {
                return v.getL1Distance(w);
            }

            
            @Override
            public double getLInfDistance(RealVector w)
                throws DimensionMismatchException {
                return v.getLInfDistance(w);
            }

            
            @Override
            public RealVector unitVector() throws MathArithmeticException {
                return v.unitVector();
            }

            
            @Override
            public void unitize() throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public RealMatrix outerProduct(RealVector w) {
                return v.outerProduct(w);
            }

            
            @Override
            public double getEntry(int index) throws OutOfRangeException {
                return v.getEntry(index);
            }

            
            @Override
            public void setEntry(int index, double value)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public void addToEntry(int index, double value)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public int getDimension() {
                return v.getDimension();
            }

            
            @Override
            public RealVector append(RealVector w) {
                return v.append(w);
            }

            
            @Override
            public RealVector append(double d) {
                return v.append(d);
            }

            
            @Override
            public RealVector getSubVector(int index, int n)
                throws OutOfRangeException, NotPositiveException {
                return v.getSubVector(index, n);
            }

            
            @Override
            public void setSubVector(int index, RealVector w)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public void set(double value)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            @Override
            public double[] toArray() {
                return v.toArray();
            }

            
            @Override
            public boolean isNaN() {
                return v.isNaN();
            }

            
            @Override
            public boolean isInfinite() {
                return v.isInfinite();
            }

            
            @Override
            public RealVector combine(double a, double b, RealVector y)
                throws DimensionMismatchException {
                return v.combine(a, b, y);
            }

            
            @Override
            public RealVector combineToSelf(double a, double b, RealVector y)
                throws MathUnsupportedOperationException {
                throw new MathUnsupportedOperationException();
            }

            
            class UnmodifiableEntry extends Entry {
                
                @Override
                public double getValue() {
                    return v.getEntry(getIndex());
                }

                
                @Override
                public void setValue(double value)
                    throws MathUnsupportedOperationException {
                    throw new MathUnsupportedOperationException();
                }
            }
        };
    }
}
