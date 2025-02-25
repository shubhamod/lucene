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
import java.util.Iterator;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class ArrayRealVector extends RealVector implements Serializable {
    
    private static final long serialVersionUID = -1097961340710804027L;
    
    private static final RealVectorFormat DEFAULT_FORMAT = RealVectorFormat.getInstance();

    
    private double data[];

    
    public ArrayRealVector() {
        data = new double[0];
    }

    
    public ArrayRealVector(int size) {
        data = new double[size];
    }

    
    public ArrayRealVector(int size, double preset) {
        data = new double[size];
        Arrays.fill(data, preset);
    }

    
    public ArrayRealVector(double[] d) {
        data = d.clone();
    }

    
    public ArrayRealVector(double[] d, boolean copyArray)
        throws NullArgumentException {
        if (d == null) {
            throw new NullArgumentException();
        }
        data = copyArray ? d.clone() :  d;
    }

    
    public ArrayRealVector(double[] d, int pos, int size)
        throws NullArgumentException, NumberIsTooLargeException {
        if (d == null) {
            throw new NullArgumentException();
        }
        if (d.length < pos + size) {
            throw new NumberIsTooLargeException(pos + size, d.length, true);
        }
        data = new double[size];
        System.arraycopy(d, pos, data, 0, size);
    }

    
    public ArrayRealVector(Double[] d) {
        data = new double[d.length];
        for (int i = 0; i < d.length; i++) {
            data[i] = d[i].doubleValue();
        }
    }

    
    public ArrayRealVector(Double[] d, int pos, int size)
        throws NullArgumentException, NumberIsTooLargeException {
        if (d == null) {
            throw new NullArgumentException();
        }
        if (d.length < pos + size) {
            throw new NumberIsTooLargeException(pos + size, d.length, true);
        }
        data = new double[size];
        for (int i = pos; i < pos + size; i++) {
            data[i - pos] = d[i].doubleValue();
        }
    }

    
    public ArrayRealVector(RealVector v) throws NullArgumentException {
        if (v == null) {
            throw new NullArgumentException();
        }
        data = new double[v.getDimension()];
        for (int i = 0; i < data.length; ++i) {
            data[i] = v.getEntry(i);
        }
    }

    
    public ArrayRealVector(ArrayRealVector v) throws NullArgumentException {
        this(v, true);
    }

    
    public ArrayRealVector(ArrayRealVector v, boolean deep) {
        data = deep ? v.data.clone() : v.data;
    }

    
    public ArrayRealVector(ArrayRealVector v1, ArrayRealVector v2) {
        data = new double[v1.data.length + v2.data.length];
        System.arraycopy(v1.data, 0, data, 0, v1.data.length);
        System.arraycopy(v2.data, 0, data, v1.data.length, v2.data.length);
    }

    
    public ArrayRealVector(ArrayRealVector v1, RealVector v2) {
        final int l1 = v1.data.length;
        final int l2 = v2.getDimension();
        data = new double[l1 + l2];
        System.arraycopy(v1.data, 0, data, 0, l1);
        for (int i = 0; i < l2; ++i) {
            data[l1 + i] = v2.getEntry(i);
        }
    }

    
    public ArrayRealVector(RealVector v1, ArrayRealVector v2) {
        final int l1 = v1.getDimension();
        final int l2 = v2.data.length;
        data = new double[l1 + l2];
        for (int i = 0; i < l1; ++i) {
            data[i] = v1.getEntry(i);
        }
        System.arraycopy(v2.data, 0, data, l1, l2);
    }

    
    public ArrayRealVector(ArrayRealVector v1, double[] v2) {
        final int l1 = v1.getDimension();
        final int l2 = v2.length;
        data = new double[l1 + l2];
        System.arraycopy(v1.data, 0, data, 0, l1);
        System.arraycopy(v2, 0, data, l1, l2);
    }

    
    public ArrayRealVector(double[] v1, ArrayRealVector v2) {
        final int l1 = v1.length;
        final int l2 = v2.getDimension();
        data = new double[l1 + l2];
        System.arraycopy(v1, 0, data, 0, l1);
        System.arraycopy(v2.data, 0, data, l1, l2);
    }

    
    public ArrayRealVector(double[] v1, double[] v2) {
        final int l1 = v1.length;
        final int l2 = v2.length;
        data = new double[l1 + l2];
        System.arraycopy(v1, 0, data, 0, l1);
        System.arraycopy(v2, 0, data, l1, l2);
    }

    
    @Override
    public ArrayRealVector copy() {
        return new ArrayRealVector(this, true);
    }

    
    @Override
    public ArrayRealVector add(RealVector v)
        throws DimensionMismatchException {
        if (v instanceof ArrayRealVector) {
            final double[] vData = ((ArrayRealVector) v).data;
            final int dim = vData.length;
            checkVectorDimensions(dim);
            ArrayRealVector result = new ArrayRealVector(dim);
            double[] resultData = result.data;
            for (int i = 0; i < dim; i++) {
                resultData[i] = data[i] + vData[i];
            }
            return result;
        } else {
            checkVectorDimensions(v);
            double[] out = data.clone();
            Iterator<Entry> it = v.iterator();
            while (it.hasNext()) {
                final Entry e = it.next();
                out[e.getIndex()] += e.getValue();
            }
            return new ArrayRealVector(out, false);
        }
    }

    
    @Override
    public ArrayRealVector subtract(RealVector v)
        throws DimensionMismatchException {
        if (v instanceof ArrayRealVector) {
            final double[] vData = ((ArrayRealVector) v).data;
            final int dim = vData.length;
            checkVectorDimensions(dim);
            ArrayRealVector result = new ArrayRealVector(dim);
            double[] resultData = result.data;
            for (int i = 0; i < dim; i++) {
                resultData[i] = data[i] - vData[i];
            }
            return result;
        } else {
            checkVectorDimensions(v);
            double[] out = data.clone();
            Iterator<Entry> it = v.iterator();
            while (it.hasNext()) {
                final Entry e = it.next();
                out[e.getIndex()] -= e.getValue();
            }
            return new ArrayRealVector(out, false);
        }
    }

    
    @Override
    public ArrayRealVector map(UnivariateFunction function) {
        return copy().mapToSelf(function);
    }

    
    @Override
    public ArrayRealVector mapToSelf(UnivariateFunction function) {
        for (int i = 0; i < data.length; i++) {
            data[i] = function.value(data[i]);
        }
        return this;
    }

    
    @Override
    public RealVector mapAddToSelf(double d) {
        for (int i = 0; i < data.length; i++) {
            data[i] += d;
        }
        return this;
    }

    
    @Override
    public RealVector mapSubtractToSelf(double d) {
        for (int i = 0; i < data.length; i++) {
            data[i] -= d;
        }
        return this;
    }

    
    @Override
    public RealVector mapMultiplyToSelf(double d) {
        for (int i = 0; i < data.length; i++) {
            data[i] *= d;
        }
        return this;
    }

    
    @Override
    public RealVector mapDivideToSelf(double d) {
        for (int i = 0; i < data.length; i++) {
            data[i] /= d;
        }
        return this;
    }

    
    @Override
    public ArrayRealVector ebeMultiply(RealVector v)
        throws DimensionMismatchException {
        if (v instanceof ArrayRealVector) {
            final double[] vData = ((ArrayRealVector) v).data;
            final int dim = vData.length;
            checkVectorDimensions(dim);
            ArrayRealVector result = new ArrayRealVector(dim);
            double[] resultData = result.data;
            for (int i = 0; i < dim; i++) {
                resultData[i] = data[i] * vData[i];
            }
            return result;
        } else {
            checkVectorDimensions(v);
            double[] out = data.clone();
            for (int i = 0; i < data.length; i++) {
                out[i] *= v.getEntry(i);
            }
            return new ArrayRealVector(out, false);
        }
    }

    
    @Override
    public ArrayRealVector ebeDivide(RealVector v)
        throws DimensionMismatchException {
        if (v instanceof ArrayRealVector) {
            final double[] vData = ((ArrayRealVector) v).data;
            final int dim = vData.length;
            checkVectorDimensions(dim);
            ArrayRealVector result = new ArrayRealVector(dim);
            double[] resultData = result.data;
            for (int i = 0; i < dim; i++) {
                resultData[i] = data[i] / vData[i];
            }
            return result;
        } else {
            checkVectorDimensions(v);
            double[] out = data.clone();
            for (int i = 0; i < data.length; i++) {
                out[i] /= v.getEntry(i);
            }
            return new ArrayRealVector(out, false);
        }
    }

    
    public double[] getDataRef() {
        return data;
    }

    
    @Override
    public double dotProduct(RealVector v) throws DimensionMismatchException {
        if (v instanceof ArrayRealVector) {
            final double[] vData = ((ArrayRealVector) v).data;
            checkVectorDimensions(vData.length);
            double dot = 0;
            for (int i = 0; i < data.length; i++) {
                dot += data[i] * vData[i];
            }
            return dot;
        }
        return super.dotProduct(v);
    }

    
    @Override
    public double getNorm() {
        double sum = 0;
        for (double a : data) {
            sum += a * a;
        }
        return FastMath.sqrt(sum);
    }

    
    @Override
    public double getL1Norm() {
        double sum = 0;
        for (double a : data) {
            sum += FastMath.abs(a);
        }
        return sum;
    }

    
    @Override
    public double getLInfNorm() {
        double max = 0;
        for (double a : data) {
            max = FastMath.max(max, FastMath.abs(a));
        }
        return max;
    }

    
    @Override
    public double getDistance(RealVector v) throws DimensionMismatchException {
        if (v instanceof ArrayRealVector) {
            final double[] vData = ((ArrayRealVector) v).data;
            checkVectorDimensions(vData.length);
            double sum = 0;
            for (int i = 0; i < data.length; ++i) {
                final double delta = data[i] - vData[i];
                sum += delta * delta;
            }
            return FastMath.sqrt(sum);
        } else {
            checkVectorDimensions(v);
            double sum = 0;
            for (int i = 0; i < data.length; ++i) {
                final double delta = data[i] - v.getEntry(i);
                sum += delta * delta;
            }
            return FastMath.sqrt(sum);
        }
    }

    
    @Override
    public double getL1Distance(RealVector v)
        throws DimensionMismatchException {
        if (v instanceof ArrayRealVector) {
            final double[] vData = ((ArrayRealVector) v).data;
            checkVectorDimensions(vData.length);
            double sum = 0;
            for (int i = 0; i < data.length; ++i) {
                final double delta = data[i] - vData[i];
                sum += FastMath.abs(delta);
            }
            return sum;
        } else {
            checkVectorDimensions(v);
            double sum = 0;
            for (int i = 0; i < data.length; ++i) {
                final double delta = data[i] - v.getEntry(i);
                sum += FastMath.abs(delta);
            }
            return sum;
        }
    }

    
    @Override
    public double getLInfDistance(RealVector v)
        throws DimensionMismatchException {
        if (v instanceof ArrayRealVector) {
            final double[] vData = ((ArrayRealVector) v).data;
            checkVectorDimensions(vData.length);
            double max = 0;
            for (int i = 0; i < data.length; ++i) {
                final double delta = data[i] - vData[i];
                max = FastMath.max(max, FastMath.abs(delta));
            }
            return max;
        } else {
            checkVectorDimensions(v);
            double max = 0;
            for (int i = 0; i < data.length; ++i) {
                final double delta = data[i] - v.getEntry(i);
                max = FastMath.max(max, FastMath.abs(delta));
            }
            return max;
        }
    }

    
    @Override
    public RealMatrix outerProduct(RealVector v) {
        if (v instanceof ArrayRealVector) {
            final double[] vData = ((ArrayRealVector) v).data;
            final int m = data.length;
            final int n = vData.length;
            final RealMatrix out = MatrixUtils.createRealMatrix(m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    out.setEntry(i, j, data[i] * vData[j]);
                }
            }
            return out;
        } else {
            final int m = data.length;
            final int n = v.getDimension();
            final RealMatrix out = MatrixUtils.createRealMatrix(m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    out.setEntry(i, j, data[i] * v.getEntry(j));
                }
            }
            return out;
        }
    }

    
    @Override
    public double getEntry(int index) throws OutOfRangeException {
        try {
            return data[index];
        } catch (IndexOutOfBoundsException e) {
            throw new OutOfRangeException(LocalizedFormats.INDEX, index, 0,
                getDimension() - 1);
        }
    }

    
    @Override
    public int getDimension() {
        return data.length;
    }

    
    @Override
    public RealVector append(RealVector v) {
        try {
            return new ArrayRealVector(this, (ArrayRealVector) v);
        } catch (ClassCastException cce) {
            return new ArrayRealVector(this, v);
        }
    }

    
    public ArrayRealVector append(ArrayRealVector v) {
        return new ArrayRealVector(this, v);
    }

    
    @Override
    public RealVector append(double in) {
        final double[] out = new double[data.length + 1];
        System.arraycopy(data, 0, out, 0, data.length);
        out[data.length] = in;
        return new ArrayRealVector(out, false);
    }

    
    @Override
    public RealVector getSubVector(int index, int n)
        throws OutOfRangeException, NotPositiveException {
        if (n < 0) {
            throw new NotPositiveException(LocalizedFormats.NUMBER_OF_ELEMENTS_SHOULD_BE_POSITIVE, n);
        }
        ArrayRealVector out = new ArrayRealVector(n);
        try {
            System.arraycopy(data, index, out.data, 0, n);
        } catch (IndexOutOfBoundsException e) {
            checkIndex(index);
            checkIndex(index + n - 1);
        }
        return out;
    }

    
    @Override
    public void setEntry(int index, double value) throws OutOfRangeException {
        try {
            data[index] = value;
        } catch (IndexOutOfBoundsException e) {
            checkIndex(index);
        }
    }

    
    @Override
    public void addToEntry(int index, double increment)
        throws OutOfRangeException {
        try {
        data[index] += increment;
        } catch(IndexOutOfBoundsException e){
            throw new OutOfRangeException(LocalizedFormats.INDEX,
                                          index, 0, data.length - 1);
        }
    }

    
    @Override
    public void setSubVector(int index, RealVector v)
        throws OutOfRangeException {
        if (v instanceof ArrayRealVector) {
            setSubVector(index, ((ArrayRealVector) v).data);
        } else {
            try {
                for (int i = index; i < index + v.getDimension(); ++i) {
                    data[i] = v.getEntry(i - index);
                }
            } catch (IndexOutOfBoundsException e) {
                checkIndex(index);
                checkIndex(index + v.getDimension() - 1);
            }
        }
    }

    
    public void setSubVector(int index, double[] v)
        throws OutOfRangeException {
        try {
            System.arraycopy(v, 0, data, index, v.length);
        } catch (IndexOutOfBoundsException e) {
            checkIndex(index);
            checkIndex(index + v.length - 1);
        }
    }

    
    @Override
    public void set(double value) {
        Arrays.fill(data, value);
    }

    
    @Override
    public double[] toArray(){
        return data.clone();
    }

    
    @Override
    public String toString(){
        return DEFAULT_FORMAT.format(this);
    }

    
    @Override
    protected void checkVectorDimensions(RealVector v)
        throws DimensionMismatchException {
        checkVectorDimensions(v.getDimension());
    }

    
    @Override
    protected void checkVectorDimensions(int n)
        throws DimensionMismatchException {
        if (data.length != n) {
            throw new DimensionMismatchException(data.length, n);
        }
    }

    
    @Override
    public boolean isNaN() {
        for (double v : data) {
            if (Double.isNaN(v)) {
                return true;
            }
        }
        return false;
    }

    
    @Override
    public boolean isInfinite() {
        if (isNaN()) {
            return false;
        }

        for (double v : data) {
            if (Double.isInfinite(v)) {
                return true;
            }
        }

        return false;
    }

    
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }

        if (!(other instanceof RealVector)) {
            return false;
        }

        RealVector rhs = (RealVector) other;
        if (data.length != rhs.getDimension()) {
            return false;
        }

        if (rhs.isNaN()) {
            return this.isNaN();
        }

        for (int i = 0; i < data.length; ++i) {
            if (data[i] != rhs.getEntry(i)) {
                return false;
            }
        }
        return true;
    }

    
    @Override
    public int hashCode() {
        if (isNaN()) {
            return 9;
        }
        return MathUtils.hash(data);
    }

    
    @Override
    public ArrayRealVector combine(double a, double b, RealVector y)
        throws DimensionMismatchException {
        return copy().combineToSelf(a, b, y);
    }

    
    @Override
    public ArrayRealVector combineToSelf(double a, double b, RealVector y)
        throws DimensionMismatchException {
        if (y instanceof ArrayRealVector) {
            final double[] yData = ((ArrayRealVector) y).data;
            checkVectorDimensions(yData.length);
            for (int i = 0; i < this.data.length; i++) {
                data[i] = a * data[i] + b * yData[i];
            }
        } else {
            checkVectorDimensions(y);
            for (int i = 0; i < this.data.length; i++) {
                data[i] = a * data[i] + b * y.getEntry(i);
            }
        }
        return this;
    }

    
    @Override
    public double walkInDefaultOrder(final RealVectorPreservingVisitor visitor) {
        visitor.start(data.length, 0, data.length - 1);
        for (int i = 0; i < data.length; i++) {
            visitor.visit(i, data[i]);
        }
        return visitor.end();
    }

    
    @Override
    public double walkInDefaultOrder(final RealVectorPreservingVisitor visitor,
        final int start, final int end) throws NumberIsTooSmallException,
        OutOfRangeException {
        checkIndices(start, end);
        visitor.start(data.length, start, end);
        for (int i = start; i <= end; i++) {
            visitor.visit(i, data[i]);
        }
        return visitor.end();
    }

    
    @Override
    public double walkInOptimizedOrder(final RealVectorPreservingVisitor visitor) {
        return walkInDefaultOrder(visitor);
    }

    
    @Override
    public double walkInOptimizedOrder(final RealVectorPreservingVisitor visitor,
        final int start, final int end) throws NumberIsTooSmallException,
        OutOfRangeException {
        return walkInDefaultOrder(visitor, start, end);
    }

    
    @Override
    public double walkInDefaultOrder(final RealVectorChangingVisitor visitor) {
        visitor.start(data.length, 0, data.length - 1);
        for (int i = 0; i < data.length; i++) {
            data[i] = visitor.visit(i, data[i]);
        }
        return visitor.end();
    }

    
    @Override
    public double walkInDefaultOrder(final RealVectorChangingVisitor visitor,
        final int start, final int end) throws NumberIsTooSmallException,
        OutOfRangeException {
        checkIndices(start, end);
        visitor.start(data.length, start, end);
        for (int i = start; i <= end; i++) {
            data[i] = visitor.visit(i, data[i]);
        }
        return visitor.end();
    }

    
    @Override
    public double walkInOptimizedOrder(final RealVectorChangingVisitor visitor) {
        return walkInDefaultOrder(visitor);
    }

    
    @Override
    public double walkInOptimizedOrder(final RealVectorChangingVisitor visitor,
        final int start, final int end) throws NumberIsTooSmallException,
        OutOfRangeException {
        return walkInDefaultOrder(visitor, start, end);
    }
}
