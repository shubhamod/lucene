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
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.OpenIntToDoubleHashMap;
import org.apache.lucene.util.hnsw.math.util.OpenIntToDoubleHashMap.Iterator;


public class OpenMapRealVector extends SparseRealVector
    implements Serializable {
    
    public static final double DEFAULT_ZERO_TOLERANCE = 1.0e-12;
    
    private static final long serialVersionUID = 8772222695580707260L;
    
    private final OpenIntToDoubleHashMap entries;
    
    private final int virtualSize;
    
    private final double epsilon;

    
    public OpenMapRealVector() {
        this(0, DEFAULT_ZERO_TOLERANCE);
    }

    
    public OpenMapRealVector(int dimension) {
        this(dimension, DEFAULT_ZERO_TOLERANCE);
    }

    
    public OpenMapRealVector(int dimension, double epsilon) {
        virtualSize = dimension;
        entries = new OpenIntToDoubleHashMap(0.0);
        this.epsilon = epsilon;
    }

    
    protected OpenMapRealVector(OpenMapRealVector v, int resize) {
        virtualSize = v.getDimension() + resize;
        entries = new OpenIntToDoubleHashMap(v.entries);
        epsilon = v.epsilon;
    }

    
    public OpenMapRealVector(int dimension, int expectedSize) {
        this(dimension, expectedSize, DEFAULT_ZERO_TOLERANCE);
    }

    
    public OpenMapRealVector(int dimension, int expectedSize, double epsilon) {
        virtualSize = dimension;
        entries = new OpenIntToDoubleHashMap(expectedSize, 0.0);
        this.epsilon = epsilon;
    }

    
    public OpenMapRealVector(double[] values) {
        this(values, DEFAULT_ZERO_TOLERANCE);
    }

    
    public OpenMapRealVector(double[] values, double epsilon) {
        virtualSize = values.length;
        entries = new OpenIntToDoubleHashMap(0.0);
        this.epsilon = epsilon;
        for (int key = 0; key < values.length; key++) {
            double value = values[key];
            if (!isDefaultValue(value)) {
                entries.put(key, value);
            }
        }
    }

    
    public OpenMapRealVector(Double[] values) {
        this(values, DEFAULT_ZERO_TOLERANCE);
    }

    
    public OpenMapRealVector(Double[] values, double epsilon) {
        virtualSize = values.length;
        entries = new OpenIntToDoubleHashMap(0.0);
        this.epsilon = epsilon;
        for (int key = 0; key < values.length; key++) {
            double value = values[key].doubleValue();
            if (!isDefaultValue(value)) {
                entries.put(key, value);
            }
        }
    }

    
    public OpenMapRealVector(OpenMapRealVector v) {
        virtualSize = v.getDimension();
        entries = new OpenIntToDoubleHashMap(v.getEntries());
        epsilon = v.epsilon;
    }

    
    public OpenMapRealVector(RealVector v) {
        virtualSize = v.getDimension();
        entries = new OpenIntToDoubleHashMap(0.0);
        epsilon = DEFAULT_ZERO_TOLERANCE;
        for (int key = 0; key < virtualSize; key++) {
            double value = v.getEntry(key);
            if (!isDefaultValue(value)) {
                entries.put(key, value);
            }
        }
    }

    
    private OpenIntToDoubleHashMap getEntries() {
        return entries;
    }

    
    protected boolean isDefaultValue(double value) {
        return FastMath.abs(value) < epsilon;
    }

    
    @Override
    public RealVector add(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        if (v instanceof OpenMapRealVector) {
            return add((OpenMapRealVector) v);
        } else {
            return super.add(v);
        }
    }

    
    public OpenMapRealVector add(OpenMapRealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        boolean copyThis = entries.size() > v.entries.size();
        OpenMapRealVector res = copyThis ? this.copy() : v.copy();
        Iterator iter = copyThis ? v.entries.iterator() : entries.iterator();
        OpenIntToDoubleHashMap randomAccess = copyThis ? entries : v.entries;
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            if (randomAccess.containsKey(key)) {
                res.setEntry(key, randomAccess.get(key) + iter.value());
            } else {
                res.setEntry(key, iter.value());
            }
        }
        return res;
    }

    
    public OpenMapRealVector append(OpenMapRealVector v) {
        OpenMapRealVector res = new OpenMapRealVector(this, v.getDimension());
        Iterator iter = v.entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            res.setEntry(iter.key() + virtualSize, iter.value());
        }
        return res;
    }

    
    @Override
    public OpenMapRealVector append(RealVector v) {
        if (v instanceof OpenMapRealVector) {
            return append((OpenMapRealVector) v);
        } else {
            final OpenMapRealVector res = new OpenMapRealVector(this, v.getDimension());
            for (int i = 0; i < v.getDimension(); i++) {
                res.setEntry(i + virtualSize, v.getEntry(i));
            }
            return res;
        }
    }

    
    @Override
    public OpenMapRealVector append(double d) {
        OpenMapRealVector res = new OpenMapRealVector(this, 1);
        res.setEntry(virtualSize, d);
        return res;
    }

    
    @Override
    public OpenMapRealVector copy() {
        return new OpenMapRealVector(this);
    }

    
    @Deprecated
    public double dotProduct(OpenMapRealVector v)
        throws DimensionMismatchException {
        return dotProduct((RealVector) v);
    }

    
    @Override
    public OpenMapRealVector ebeDivide(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        OpenMapRealVector res = new OpenMapRealVector(this);
        /*
         * MATH-803: it is not sufficient to loop through non zero entries of
         * this only. Indeed, if this[i] = 0d and v[i] = 0d, then
         * this[i] / v[i] = NaN, and not 0d.
         */
        final int n = getDimension();
        for (int i = 0; i < n; i++) {
            res.setEntry(i, this.getEntry(i) / v.getEntry(i));
        }
        return res;
    }

    
    @Override
    public OpenMapRealVector ebeMultiply(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        OpenMapRealVector res = new OpenMapRealVector(this);
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            res.setEntry(iter.key(), iter.value() * v.getEntry(iter.key()));
        }
        return res;
    }

    
    @Override
    public OpenMapRealVector getSubVector(int index, int n)
        throws NotPositiveException, OutOfRangeException {
        checkIndex(index);
        if (n < 0) {
            throw new NotPositiveException(LocalizedFormats.NUMBER_OF_ELEMENTS_SHOULD_BE_POSITIVE, n);
        }
        checkIndex(index + n - 1);
        OpenMapRealVector res = new OpenMapRealVector(n);
        int end = index + n;
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            if (key >= index && key < end) {
                res.setEntry(key - index, iter.value());
            }
        }
        return res;
    }

    
    @Override
    public int getDimension() {
        return virtualSize;
    }

    
    public double getDistance(OpenMapRealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        Iterator iter = entries.iterator();
        double res = 0;
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            double delta;
            delta = iter.value() - v.getEntry(key);
            res += delta * delta;
        }
        iter = v.getEntries().iterator();
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            if (!entries.containsKey(key)) {
                final double value = iter.value();
                res += value * value;
            }
        }
        return FastMath.sqrt(res);
    }

    
    @Override
    public double getDistance(RealVector v) throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        if (v instanceof OpenMapRealVector) {
            return getDistance((OpenMapRealVector) v);
        } else {
            return super.getDistance(v);
        }
    }

    
    @Override
    public double getEntry(int index) throws OutOfRangeException {
        checkIndex(index);
        return entries.get(index);
    }

    
    public double getL1Distance(OpenMapRealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        double max = 0;
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            double delta = FastMath.abs(iter.value() - v.getEntry(iter.key()));
            max += delta;
        }
        iter = v.getEntries().iterator();
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            if (!entries.containsKey(key)) {
                double delta = FastMath.abs(iter.value());
                max +=  FastMath.abs(delta);
            }
        }
        return max;
    }

    
    @Override
    public double getL1Distance(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        if (v instanceof OpenMapRealVector) {
            return getL1Distance((OpenMapRealVector) v);
        } else {
            return super.getL1Distance(v);
        }
    }

    
    private double getLInfDistance(OpenMapRealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        double max = 0;
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            double delta = FastMath.abs(iter.value() - v.getEntry(iter.key()));
            if (delta > max) {
                max = delta;
            }
        }
        iter = v.getEntries().iterator();
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            if (!entries.containsKey(key) && iter.value() > max) {
                max = iter.value();
            }
        }
        return max;
    }

    
    @Override
    public double getLInfDistance(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        if (v instanceof OpenMapRealVector) {
            return getLInfDistance((OpenMapRealVector) v);
        } else {
            return super.getLInfDistance(v);
        }
    }

    
    @Override
    public boolean isInfinite() {
        boolean infiniteFound = false;
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            final double value = iter.value();
            if (Double.isNaN(value)) {
                return false;
            }
            if (Double.isInfinite(value)) {
                infiniteFound = true;
            }
        }
        return infiniteFound;
    }

    
    @Override
    public boolean isNaN() {
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            if (Double.isNaN(iter.value())) {
                return true;
            }
        }
        return false;
    }

    
    @Override
    public OpenMapRealVector mapAdd(double d) {
        return copy().mapAddToSelf(d);
    }

    
    @Override
    public OpenMapRealVector mapAddToSelf(double d) {
        for (int i = 0; i < virtualSize; i++) {
            setEntry(i, getEntry(i) + d);
        }
        return this;
    }

    
    @Override
    public void setEntry(int index, double value)
        throws OutOfRangeException {
        checkIndex(index);
        if (!isDefaultValue(value)) {
            entries.put(index, value);
        } else if (entries.containsKey(index)) {
            entries.remove(index);
        }
    }

    
    @Override
    public void setSubVector(int index, RealVector v)
        throws OutOfRangeException {
        checkIndex(index);
        checkIndex(index + v.getDimension() - 1);
        for (int i = 0; i < v.getDimension(); i++) {
            setEntry(i + index, v.getEntry(i));
        }
    }

    
    @Override
    public void set(double value) {
        for (int i = 0; i < virtualSize; i++) {
            setEntry(i, value);
        }
    }

    
    public OpenMapRealVector subtract(OpenMapRealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        OpenMapRealVector res = copy();
        Iterator iter = v.getEntries().iterator();
        while (iter.hasNext()) {
            iter.advance();
            int key = iter.key();
            if (entries.containsKey(key)) {
                res.setEntry(key, entries.get(key) - iter.value());
            } else {
                res.setEntry(key, -iter.value());
            }
        }
        return res;
    }

    
    @Override
    public RealVector subtract(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
        if (v instanceof OpenMapRealVector) {
            return subtract((OpenMapRealVector) v);
        } else {
            return super.subtract(v);
        }
    }

    
    @Override
    public OpenMapRealVector unitVector() throws MathArithmeticException {
        OpenMapRealVector res = copy();
        res.unitize();
        return res;
    }

    
    @Override
    public void unitize() throws MathArithmeticException {
        double norm = getNorm();
        if (isDefaultValue(norm)) {
            throw new MathArithmeticException(LocalizedFormats.ZERO_NORM);
        }
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            entries.put(iter.key(), iter.value() / norm);
        }
    }

    
    @Override
    public double[] toArray() {
        double[] res = new double[virtualSize];
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            res[iter.key()] = iter.value();
        }
        return res;
    }

    
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        long temp;
        temp = Double.doubleToLongBits(epsilon);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        result = prime * result + virtualSize;
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            temp = Double.doubleToLongBits(iter.value());
            result = prime * result + (int) (temp ^ (temp >>32));
        }
        return result;
    }

    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof OpenMapRealVector)) {
            return false;
        }
        OpenMapRealVector other = (OpenMapRealVector) obj;
        if (virtualSize != other.virtualSize) {
            return false;
        }
        if (Double.doubleToLongBits(epsilon) !=
            Double.doubleToLongBits(other.epsilon)) {
            return false;
        }
        Iterator iter = entries.iterator();
        while (iter.hasNext()) {
            iter.advance();
            double test = other.getEntry(iter.key());
            if (Double.doubleToLongBits(test) != Double.doubleToLongBits(iter.value())) {
                return false;
            }
        }
        iter = other.getEntries().iterator();
        while (iter.hasNext()) {
            iter.advance();
            double test = iter.value();
            if (Double.doubleToLongBits(test) != Double.doubleToLongBits(getEntry(iter.key()))) {
                return false;
            }
        }
        return true;
    }

    
    public double getSparsity() {
        return (double)entries.size()/(double)getDimension();
    }

    
    @Override
    public java.util.Iterator<Entry> sparseIterator() {
        return new OpenMapSparseIterator();
    }

    
    protected class OpenMapEntry extends Entry {
        
        private final Iterator iter;

        
        protected OpenMapEntry(Iterator iter) {
            this.iter = iter;
        }

        
        @Override
        public double getValue() {
            return iter.value();
        }

        
        @Override
        public void setValue(double value) {
            entries.put(iter.key(), value);
        }

        
        @Override
        public int getIndex() {
            return iter.key();
        }

    }

    
    protected class OpenMapSparseIterator implements java.util.Iterator<Entry> {
        
        private final Iterator iter;
        
        private final Entry current;

        
        protected OpenMapSparseIterator() {
            iter = entries.iterator();
            current = new OpenMapEntry(iter);
        }

        
        public boolean hasNext() {
            return iter.hasNext();
        }

        
        public Entry next() {
            iter.advance();
            return current;
        }

        
        public void remove() {
            throw new UnsupportedOperationException("Not supported");
        }
    }
}
