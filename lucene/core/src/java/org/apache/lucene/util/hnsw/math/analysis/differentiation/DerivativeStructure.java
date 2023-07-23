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
package org.apache.lucene.util.hnsw.math.analysis.differentiation;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class DerivativeStructure implements RealFieldElement<DerivativeStructure>, Serializable {

    
    private static final long serialVersionUID = 20120730L;

    
    private transient DSCompiler compiler;

    
    private final double[] data;

    
    private DerivativeStructure(final DSCompiler compiler) {
        this.compiler = compiler;
        this.data     = new double[compiler.getSize()];
    }

    
    public DerivativeStructure(final int parameters, final int order)
        throws NumberIsTooLargeException {
        this(DSCompiler.getCompiler(parameters, order));
    }

    
    public DerivativeStructure(final int parameters, final int order, final double value)
        throws NumberIsTooLargeException {
        this(parameters, order);
        this.data[0] = value;
    }

    
    public DerivativeStructure(final int parameters, final int order,
                               final int index, final double value)
        throws NumberIsTooLargeException {
        this(parameters, order, value);

        if (index >= parameters) {
            throw new NumberIsTooLargeException(index, parameters, false);
        }

        if (order > 0) {
            // the derivative of the variable with respect to itself is 1.
            data[DSCompiler.getCompiler(index, order).getSize()] = 1.0;
        }

    }

    
    public DerivativeStructure(final double a1, final DerivativeStructure ds1,
                               final double a2, final DerivativeStructure ds2)
        throws DimensionMismatchException {
        this(ds1.compiler);
        compiler.checkCompatibility(ds2.compiler);
        compiler.linearCombination(a1, ds1.data, 0, a2, ds2.data, 0, data, 0);
    }

    
    public DerivativeStructure(final double a1, final DerivativeStructure ds1,
                               final double a2, final DerivativeStructure ds2,
                               final double a3, final DerivativeStructure ds3)
        throws DimensionMismatchException {
        this(ds1.compiler);
        compiler.checkCompatibility(ds2.compiler);
        compiler.checkCompatibility(ds3.compiler);
        compiler.linearCombination(a1, ds1.data, 0, a2, ds2.data, 0, a3, ds3.data, 0, data, 0);
    }

    
    public DerivativeStructure(final double a1, final DerivativeStructure ds1,
                               final double a2, final DerivativeStructure ds2,
                               final double a3, final DerivativeStructure ds3,
                               final double a4, final DerivativeStructure ds4)
        throws DimensionMismatchException {
        this(ds1.compiler);
        compiler.checkCompatibility(ds2.compiler);
        compiler.checkCompatibility(ds3.compiler);
        compiler.checkCompatibility(ds4.compiler);
        compiler.linearCombination(a1, ds1.data, 0, a2, ds2.data, 0,
                                   a3, ds3.data, 0, a4, ds4.data, 0,
                                   data, 0);
    }

    
    public DerivativeStructure(final int parameters, final int order, final double ... derivatives)
        throws DimensionMismatchException, NumberIsTooLargeException {
        this(parameters, order);
        if (derivatives.length != data.length) {
            throw new DimensionMismatchException(derivatives.length, data.length);
        }
        System.arraycopy(derivatives, 0, data, 0, data.length);
    }

    
    private DerivativeStructure(final DerivativeStructure ds) {
        this.compiler = ds.compiler;
        this.data     = ds.data.clone();
    }

    
    public int getFreeParameters() {
        return compiler.getFreeParameters();
    }

    
    public int getOrder() {
        return compiler.getOrder();
    }

    
    public DerivativeStructure createConstant(final double c) {
        return new DerivativeStructure(getFreeParameters(), getOrder(), c);
    }

    
    public double getReal() {
        return data[0];
    }

    
    public double getValue() {
        return data[0];
    }

    
    public double getPartialDerivative(final int ... orders)
        throws DimensionMismatchException, NumberIsTooLargeException {
        return data[compiler.getPartialDerivativeIndex(orders)];
    }

    
    public double[] getAllDerivatives() {
        return data.clone();
    }

    
    public DerivativeStructure add(final double a) {
        final DerivativeStructure ds = new DerivativeStructure(this);
        ds.data[0] += a;
        return ds;
    }

    
    public DerivativeStructure add(final DerivativeStructure a)
        throws DimensionMismatchException {
        compiler.checkCompatibility(a.compiler);
        final DerivativeStructure ds = new DerivativeStructure(this);
        compiler.add(data, 0, a.data, 0, ds.data, 0);
        return ds;
    }

    
    public DerivativeStructure subtract(final double a) {
        return add(-a);
    }

    
    public DerivativeStructure subtract(final DerivativeStructure a)
        throws DimensionMismatchException {
        compiler.checkCompatibility(a.compiler);
        final DerivativeStructure ds = new DerivativeStructure(this);
        compiler.subtract(data, 0, a.data, 0, ds.data, 0);
        return ds;
    }

    
    public DerivativeStructure multiply(final int n) {
        return multiply((double) n);
    }

    
    public DerivativeStructure multiply(final double a) {
        final DerivativeStructure ds = new DerivativeStructure(this);
        for (int i = 0; i < ds.data.length; ++i) {
            ds.data[i] *= a;
        }
        return ds;
    }

    
    public DerivativeStructure multiply(final DerivativeStructure a)
        throws DimensionMismatchException {
        compiler.checkCompatibility(a.compiler);
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.multiply(data, 0, a.data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure divide(final double a) {
        final DerivativeStructure ds = new DerivativeStructure(this);
        for (int i = 0; i < ds.data.length; ++i) {
            ds.data[i] /= a;
        }
        return ds;
    }

    
    public DerivativeStructure divide(final DerivativeStructure a)
        throws DimensionMismatchException {
        compiler.checkCompatibility(a.compiler);
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.divide(data, 0, a.data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure remainder(final double a) {
        final DerivativeStructure ds = new DerivativeStructure(this);
        ds.data[0] = FastMath.IEEEremainder(ds.data[0], a);
        return ds;
    }

    
    public DerivativeStructure remainder(final DerivativeStructure a)
        throws DimensionMismatchException {
        compiler.checkCompatibility(a.compiler);
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.remainder(data, 0, a.data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure negate() {
        final DerivativeStructure ds = new DerivativeStructure(compiler);
        for (int i = 0; i < ds.data.length; ++i) {
            ds.data[i] = -data[i];
        }
        return ds;
    }

    
    public DerivativeStructure abs() {
        if (Double.doubleToLongBits(data[0]) < 0) {
            // we use the bits representation to also handle -0.0
            return negate();
        } else {
            return this;
        }
    }

    
    public DerivativeStructure ceil() {
        return new DerivativeStructure(compiler.getFreeParameters(),
                                       compiler.getOrder(),
                                       FastMath.ceil(data[0]));
    }

    
    public DerivativeStructure floor() {
        return new DerivativeStructure(compiler.getFreeParameters(),
                                       compiler.getOrder(),
                                       FastMath.floor(data[0]));
    }

    
    public DerivativeStructure rint() {
        return new DerivativeStructure(compiler.getFreeParameters(),
                                       compiler.getOrder(),
                                       FastMath.rint(data[0]));
    }

    
    public long round() {
        return FastMath.round(data[0]);
    }

    
    public DerivativeStructure signum() {
        return new DerivativeStructure(compiler.getFreeParameters(),
                                       compiler.getOrder(),
                                       FastMath.signum(data[0]));
    }

    
    public DerivativeStructure copySign(final DerivativeStructure sign){
        long m = Double.doubleToLongBits(data[0]);
        long s = Double.doubleToLongBits(sign.data[0]);
        if ((m >= 0 && s >= 0) || (m < 0 && s < 0)) { // Sign is currently OK
            return this;
        }
        return negate(); // flip sign
    }

    
    public DerivativeStructure copySign(final double sign) {
        long m = Double.doubleToLongBits(data[0]);
        long s = Double.doubleToLongBits(sign);
        if ((m >= 0 && s >= 0) || (m < 0 && s < 0)) { // Sign is currently OK
            return this;
        }
        return negate(); // flip sign
    }

    
    public int getExponent() {
        return FastMath.getExponent(data[0]);
    }

    
    public DerivativeStructure scalb(final int n) {
        final DerivativeStructure ds = new DerivativeStructure(compiler);
        for (int i = 0; i < ds.data.length; ++i) {
            ds.data[i] = FastMath.scalb(data[i], n);
        }
        return ds;
    }

    
    public DerivativeStructure hypot(final DerivativeStructure y)
        throws DimensionMismatchException {

        compiler.checkCompatibility(y.compiler);

        if (Double.isInfinite(data[0]) || Double.isInfinite(y.data[0])) {
            return new DerivativeStructure(compiler.getFreeParameters(),
                                           compiler.getFreeParameters(),
                                           Double.POSITIVE_INFINITY);
        } else if (Double.isNaN(data[0]) || Double.isNaN(y.data[0])) {
            return new DerivativeStructure(compiler.getFreeParameters(),
                                           compiler.getFreeParameters(),
                                           Double.NaN);
        } else {

            final int expX = getExponent();
            final int expY = y.getExponent();
            if (expX > expY + 27) {
                // y is neglectible with respect to x
                return abs();
            } else if (expY > expX + 27) {
                // x is neglectible with respect to y
                return y.abs();
            } else {

                // find an intermediate scale to avoid both overflow and underflow
                final int middleExp = (expX + expY) / 2;

                // scale parameters without losing precision
                final DerivativeStructure scaledX = scalb(-middleExp);
                final DerivativeStructure scaledY = y.scalb(-middleExp);

                // compute scaled hypotenuse
                final DerivativeStructure scaledH =
                        scaledX.multiply(scaledX).add(scaledY.multiply(scaledY)).sqrt();

                // remove scaling
                return scaledH.scalb(middleExp);

            }

        }
    }

    
    public static DerivativeStructure hypot(final DerivativeStructure x, final DerivativeStructure y)
        throws DimensionMismatchException {
        return x.hypot(y);
    }

    
    public DerivativeStructure compose(final double ... f)
        throws DimensionMismatchException {
        if (f.length != getOrder() + 1) {
            throw new DimensionMismatchException(f.length, getOrder() + 1);
        }
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.compose(data, 0, f, result.data, 0);
        return result;
    }

    
    public DerivativeStructure reciprocal() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.pow(data, 0, -1, result.data, 0);
        return result;
    }

    
    public DerivativeStructure sqrt() {
        return rootN(2);
    }

    
    public DerivativeStructure cbrt() {
        return rootN(3);
    }

    
    public DerivativeStructure rootN(final int n) {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.rootN(data, 0, n, result.data, 0);
        return result;
    }

    
    public Field<DerivativeStructure> getField() {
        return new Field<DerivativeStructure>() {

            
            public DerivativeStructure getZero() {
                return new DerivativeStructure(compiler.getFreeParameters(), compiler.getOrder(), 0.0);
            }

            
            public DerivativeStructure getOne() {
                return new DerivativeStructure(compiler.getFreeParameters(), compiler.getOrder(), 1.0);
            }

            
            public Class<? extends FieldElement<DerivativeStructure>> getRuntimeClass() {
                return DerivativeStructure.class;
            }

        };
    }

    
    public static DerivativeStructure pow(final double a, final DerivativeStructure x) {
        final DerivativeStructure result = new DerivativeStructure(x.compiler);
        x.compiler.pow(a, x.data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure pow(final double p) {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.pow(data, 0, p, result.data, 0);
        return result;
    }

    
    public DerivativeStructure pow(final int n) {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.pow(data, 0, n, result.data, 0);
        return result;
    }

    
    public DerivativeStructure pow(final DerivativeStructure e)
        throws DimensionMismatchException {
        compiler.checkCompatibility(e.compiler);
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.pow(data, 0, e.data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure exp() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.exp(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure expm1() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.expm1(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure log() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.log(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure log1p() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.log1p(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure log10() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.log10(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure cos() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.cos(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure sin() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.sin(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure tan() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.tan(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure acos() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.acos(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure asin() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.asin(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure atan() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.atan(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure atan2(final DerivativeStructure x)
        throws DimensionMismatchException {
        compiler.checkCompatibility(x.compiler);
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.atan2(data, 0, x.data, 0, result.data, 0);
        return result;
    }

    
    public static DerivativeStructure atan2(final DerivativeStructure y, final DerivativeStructure x)
        throws DimensionMismatchException {
        return y.atan2(x);
    }

    
    public DerivativeStructure cosh() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.cosh(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure sinh() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.sinh(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure tanh() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.tanh(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure acosh() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.acosh(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure asinh() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.asinh(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure atanh() {
        final DerivativeStructure result = new DerivativeStructure(compiler);
        compiler.atanh(data, 0, result.data, 0);
        return result;
    }

    
    public DerivativeStructure toDegrees() {
        final DerivativeStructure ds = new DerivativeStructure(compiler);
        for (int i = 0; i < ds.data.length; ++i) {
            ds.data[i] = FastMath.toDegrees(data[i]);
        }
        return ds;
    }

    
    public DerivativeStructure toRadians() {
        final DerivativeStructure ds = new DerivativeStructure(compiler);
        for (int i = 0; i < ds.data.length; ++i) {
            ds.data[i] = FastMath.toRadians(data[i]);
        }
        return ds;
    }

    
    public double taylor(final double ... delta) throws MathArithmeticException {
        return compiler.taylor(data, 0, delta);
    }

    
    public DerivativeStructure linearCombination(final DerivativeStructure[] a, final DerivativeStructure[] b)
        throws DimensionMismatchException {

        // compute an accurate value, taking care of cancellations
        final double[] aDouble = new double[a.length];
        for (int i = 0; i < a.length; ++i) {
            aDouble[i] = a[i].getValue();
        }
        final double[] bDouble = new double[b.length];
        for (int i = 0; i < b.length; ++i) {
            bDouble[i] = b[i].getValue();
        }
        final double accurateValue = MathArrays.linearCombination(aDouble, bDouble);

        // compute a simple value, with all partial derivatives
        DerivativeStructure simpleValue = a[0].getField().getZero();
        for (int i = 0; i < a.length; ++i) {
            simpleValue = simpleValue.add(a[i].multiply(b[i]));
        }

        // create a result with accurate value and all derivatives (not necessarily as accurate as the value)
        final double[] all = simpleValue.getAllDerivatives();
        all[0] = accurateValue;
        return new DerivativeStructure(simpleValue.getFreeParameters(), simpleValue.getOrder(), all);

    }

    
    public DerivativeStructure linearCombination(final double[] a, final DerivativeStructure[] b)
        throws DimensionMismatchException {

        // compute an accurate value, taking care of cancellations
        final double[] bDouble = new double[b.length];
        for (int i = 0; i < b.length; ++i) {
            bDouble[i] = b[i].getValue();
        }
        final double accurateValue = MathArrays.linearCombination(a, bDouble);

        // compute a simple value, with all partial derivatives
        DerivativeStructure simpleValue = b[0].getField().getZero();
        for (int i = 0; i < a.length; ++i) {
            simpleValue = simpleValue.add(b[i].multiply(a[i]));
        }

        // create a result with accurate value and all derivatives (not necessarily as accurate as the value)
        final double[] all = simpleValue.getAllDerivatives();
        all[0] = accurateValue;
        return new DerivativeStructure(simpleValue.getFreeParameters(), simpleValue.getOrder(), all);

    }

    
    public DerivativeStructure linearCombination(final DerivativeStructure a1, final DerivativeStructure b1,
                                                 final DerivativeStructure a2, final DerivativeStructure b2)
        throws DimensionMismatchException {

        // compute an accurate value, taking care of cancellations
        final double accurateValue = MathArrays.linearCombination(a1.getValue(), b1.getValue(),
                                                                  a2.getValue(), b2.getValue());

        // compute a simple value, with all partial derivatives
        final DerivativeStructure simpleValue = a1.multiply(b1).add(a2.multiply(b2));

        // create a result with accurate value and all derivatives (not necessarily as accurate as the value)
        final double[] all = simpleValue.getAllDerivatives();
        all[0] = accurateValue;
        return new DerivativeStructure(getFreeParameters(), getOrder(), all);

    }

    
    public DerivativeStructure linearCombination(final double a1, final DerivativeStructure b1,
                                                 final double a2, final DerivativeStructure b2)
        throws DimensionMismatchException {

        // compute an accurate value, taking care of cancellations
        final double accurateValue = MathArrays.linearCombination(a1, b1.getValue(),
                                                                  a2, b2.getValue());

        // compute a simple value, with all partial derivatives
        final DerivativeStructure simpleValue = b1.multiply(a1).add(b2.multiply(a2));

        // create a result with accurate value and all derivatives (not necessarily as accurate as the value)
        final double[] all = simpleValue.getAllDerivatives();
        all[0] = accurateValue;
        return new DerivativeStructure(getFreeParameters(), getOrder(), all);

    }

    
    public DerivativeStructure linearCombination(final DerivativeStructure a1, final DerivativeStructure b1,
                                                 final DerivativeStructure a2, final DerivativeStructure b2,
                                                 final DerivativeStructure a3, final DerivativeStructure b3)
        throws DimensionMismatchException {

        // compute an accurate value, taking care of cancellations
        final double accurateValue = MathArrays.linearCombination(a1.getValue(), b1.getValue(),
                                                                  a2.getValue(), b2.getValue(),
                                                                  a3.getValue(), b3.getValue());

        // compute a simple value, with all partial derivatives
        final DerivativeStructure simpleValue = a1.multiply(b1).add(a2.multiply(b2)).add(a3.multiply(b3));

        // create a result with accurate value and all derivatives (not necessarily as accurate as the value)
        final double[] all = simpleValue.getAllDerivatives();
        all[0] = accurateValue;
        return new DerivativeStructure(getFreeParameters(), getOrder(), all);

    }

    
    public DerivativeStructure linearCombination(final double a1, final DerivativeStructure b1,
                                                 final double a2, final DerivativeStructure b2,
                                                 final double a3, final DerivativeStructure b3)
        throws DimensionMismatchException {

        // compute an accurate value, taking care of cancellations
        final double accurateValue = MathArrays.linearCombination(a1, b1.getValue(),
                                                                  a2, b2.getValue(),
                                                                  a3, b3.getValue());

        // compute a simple value, with all partial derivatives
        final DerivativeStructure simpleValue = b1.multiply(a1).add(b2.multiply(a2)).add(b3.multiply(a3));

        // create a result with accurate value and all derivatives (not necessarily as accurate as the value)
        final double[] all = simpleValue.getAllDerivatives();
        all[0] = accurateValue;
        return new DerivativeStructure(getFreeParameters(), getOrder(), all);

    }

    
    public DerivativeStructure linearCombination(final DerivativeStructure a1, final DerivativeStructure b1,
                                                 final DerivativeStructure a2, final DerivativeStructure b2,
                                                 final DerivativeStructure a3, final DerivativeStructure b3,
                                                 final DerivativeStructure a4, final DerivativeStructure b4)
        throws DimensionMismatchException {

        // compute an accurate value, taking care of cancellations
        final double accurateValue = MathArrays.linearCombination(a1.getValue(), b1.getValue(),
                                                                  a2.getValue(), b2.getValue(),
                                                                  a3.getValue(), b3.getValue(),
                                                                  a4.getValue(), b4.getValue());

        // compute a simple value, with all partial derivatives
        final DerivativeStructure simpleValue = a1.multiply(b1).add(a2.multiply(b2)).add(a3.multiply(b3)).add(a4.multiply(b4));

        // create a result with accurate value and all derivatives (not necessarily as accurate as the value)
        final double[] all = simpleValue.getAllDerivatives();
        all[0] = accurateValue;
        return new DerivativeStructure(getFreeParameters(), getOrder(), all);

    }

    
    public DerivativeStructure linearCombination(final double a1, final DerivativeStructure b1,
                                                 final double a2, final DerivativeStructure b2,
                                                 final double a3, final DerivativeStructure b3,
                                                 final double a4, final DerivativeStructure b4)
        throws DimensionMismatchException {

        // compute an accurate value, taking care of cancellations
        final double accurateValue = MathArrays.linearCombination(a1, b1.getValue(),
                                                                  a2, b2.getValue(),
                                                                  a3, b3.getValue(),
                                                                  a4, b4.getValue());

        // compute a simple value, with all partial derivatives
        final DerivativeStructure simpleValue = b1.multiply(a1).add(b2.multiply(a2)).add(b3.multiply(a3)).add(b4.multiply(a4));

        // create a result with accurate value and all derivatives (not necessarily as accurate as the value)
        final double[] all = simpleValue.getAllDerivatives();
        all[0] = accurateValue;
        return new DerivativeStructure(getFreeParameters(), getOrder(), all);

    }

    
    @Override
    public boolean equals(Object other) {

        if (this == other) {
            return true;
        }

        if (other instanceof DerivativeStructure) {
            final DerivativeStructure rhs = (DerivativeStructure)other;
            return (getFreeParameters() == rhs.getFreeParameters()) &&
                   (getOrder() == rhs.getOrder()) &&
                   MathArrays.equals(data, rhs.data);
        }

        return false;

    }

    
    @Override
    public int hashCode() {
        return 227 + 229 * getFreeParameters() + 233 * getOrder() + 239 * MathUtils.hash(data);
    }

    
    private Object writeReplace() {
        return new DataTransferObject(compiler.getFreeParameters(), compiler.getOrder(), data);
    }

    
    private static class DataTransferObject implements Serializable {

        
        private static final long serialVersionUID = 20120730L;

        
        private final int variables;

        
        private final int order;

        
        private final double[] data;

        
        DataTransferObject(final int variables, final int order, final double[] data) {
            this.variables = variables;
            this.order     = order;
            this.data      = data;
        }

        
        private Object readResolve() {
            return new DerivativeStructure(variables, order, data);
        }

    }

}
