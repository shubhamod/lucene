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
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.RealFieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class SparseGradient implements RealFieldElement<SparseGradient>, Serializable {

    
    private static final long serialVersionUID = 20131025L;

    
    private double value;

    
    private final Map<Integer, Double> derivatives;

    
    private SparseGradient(final double value, final Map<Integer, Double> derivatives) {
        this.value = value;
        this.derivatives = new HashMap<Integer, Double>();
        if (derivatives != null) {
            this.derivatives.putAll(derivatives);
        }
    }

    
    private SparseGradient(final double value, final double scale,
                             final Map<Integer, Double> derivatives) {
        this.value = value;
        this.derivatives = new HashMap<Integer, Double>();
        if (derivatives != null) {
            for (final Map.Entry<Integer, Double> entry : derivatives.entrySet()) {
                this.derivatives.put(entry.getKey(), scale * entry.getValue());
            }
        }
    }

    
    public static SparseGradient createConstant(final double value) {
        return new SparseGradient(value, Collections.<Integer, Double> emptyMap());
    }

    
    public static SparseGradient createVariable(final int idx, final double value) {
        return new SparseGradient(value, Collections.singletonMap(idx, 1.0));
    }

    
    public int numVars() {
        return derivatives.size();
    }

    
    public double getDerivative(final int index) {
        final Double out = derivatives.get(index);
        return (out == null) ? 0.0 : out;
    }

    
    public double getValue() {
        return value;
    }

    
    public double getReal() {
        return value;
    }

    
    public SparseGradient add(final SparseGradient a) {
        final SparseGradient out = new SparseGradient(value + a.value, derivatives);
        for (Map.Entry<Integer, Double> entry : a.derivatives.entrySet()) {
            final int id = entry.getKey();
            final Double old = out.derivatives.get(id);
            if (old == null) {
                out.derivatives.put(id, entry.getValue());
            } else {
                out.derivatives.put(id, old + entry.getValue());
            }
        }

        return out;
    }

    
    public void addInPlace(final SparseGradient a) {
        value += a.value;
        for (final Map.Entry<Integer, Double> entry : a.derivatives.entrySet()) {
            final int id = entry.getKey();
            final Double old = derivatives.get(id);
            if (old == null) {
                derivatives.put(id, entry.getValue());
            } else {
                derivatives.put(id, old + entry.getValue());
            }
        }
    }

    
    public SparseGradient add(final double c) {
        final SparseGradient out = new SparseGradient(value + c, derivatives);
        return out;
    }

    
    public SparseGradient subtract(final SparseGradient a) {
        final SparseGradient out = new SparseGradient(value - a.value, derivatives);
        for (Map.Entry<Integer, Double> entry : a.derivatives.entrySet()) {
            final int id = entry.getKey();
            final Double old = out.derivatives.get(id);
            if (old == null) {
                out.derivatives.put(id, -entry.getValue());
            } else {
                out.derivatives.put(id, old - entry.getValue());
            }
        }
        return out;
    }

    
    public SparseGradient subtract(double c) {
        return new SparseGradient(value - c, derivatives);
    }

    
    public SparseGradient multiply(final SparseGradient a) {
        final SparseGradient out =
            new SparseGradient(value * a.value, Collections.<Integer, Double> emptyMap());

        // Derivatives.
        for (Map.Entry<Integer, Double> entry : derivatives.entrySet()) {
            out.derivatives.put(entry.getKey(), a.value * entry.getValue());
        }
        for (Map.Entry<Integer, Double> entry : a.derivatives.entrySet()) {
            final int id = entry.getKey();
            final Double old = out.derivatives.get(id);
            if (old == null) {
                out.derivatives.put(id, value * entry.getValue());
            } else {
                out.derivatives.put(id, old + value * entry.getValue());
            }
        }
        return out;
    }

    
    public void multiplyInPlace(final SparseGradient a) {
        // Derivatives.
        for (Map.Entry<Integer, Double> entry : derivatives.entrySet()) {
            derivatives.put(entry.getKey(), a.value * entry.getValue());
        }
        for (Map.Entry<Integer, Double> entry : a.derivatives.entrySet()) {
            final int id = entry.getKey();
            final Double old = derivatives.get(id);
            if (old == null) {
                derivatives.put(id, value * entry.getValue());
            } else {
                derivatives.put(id, old + value * entry.getValue());
            }
        }
        value *= a.value;
    }

    
    public SparseGradient multiply(final double c) {
        return new SparseGradient(value * c, c, derivatives);
    }

    
    public SparseGradient multiply(final int n) {
        return new SparseGradient(value * n, n, derivatives);
    }

    
    public SparseGradient divide(final SparseGradient a) {
        final SparseGradient out = new SparseGradient(value / a.value, Collections.<Integer, Double> emptyMap());

        // Derivatives.
        for (Map.Entry<Integer, Double> entry : derivatives.entrySet()) {
            out.derivatives.put(entry.getKey(), entry.getValue() / a.value);
        }
        for (Map.Entry<Integer, Double> entry : a.derivatives.entrySet()) {
            final int id = entry.getKey();
            final Double old = out.derivatives.get(id);
            if (old == null) {
                out.derivatives.put(id, -out.value / a.value * entry.getValue());
            } else {
                out.derivatives.put(id, old - out.value / a.value * entry.getValue());
            }
        }
        return out;
    }

    
    public SparseGradient divide(final double c) {
        return new SparseGradient(value / c, 1.0 / c, derivatives);
    }

    
    public SparseGradient negate() {
        return new SparseGradient(-value, -1.0, derivatives);
    }

    
    public Field<SparseGradient> getField() {
        return new Field<SparseGradient>() {

            
            public SparseGradient getZero() {
                return createConstant(0);
            }

            
            public SparseGradient getOne() {
                return createConstant(1);
            }

            
            public Class<? extends FieldElement<SparseGradient>> getRuntimeClass() {
                return SparseGradient.class;
            }

        };
    }

    
    public SparseGradient remainder(final double a) {
        return new SparseGradient(FastMath.IEEEremainder(value, a), derivatives);
    }

    
    public SparseGradient remainder(final SparseGradient a) {

        // compute k such that lhs % rhs = lhs - k rhs
        final double rem = FastMath.IEEEremainder(value, a.value);
        final double k   = FastMath.rint((value - rem) / a.value);

        return subtract(a.multiply(k));

    }

    
    public SparseGradient abs() {
        if (Double.doubleToLongBits(value) < 0) {
            // we use the bits representation to also handle -0.0
            return negate();
        } else {
            return this;
        }
    }

    
    public SparseGradient ceil() {
        return createConstant(FastMath.ceil(value));
    }

    
    public SparseGradient floor() {
        return createConstant(FastMath.floor(value));
    }

    
    public SparseGradient rint() {
        return createConstant(FastMath.rint(value));
    }

    
    public long round() {
        return FastMath.round(value);
    }

    
    public SparseGradient signum() {
        return createConstant(FastMath.signum(value));
    }

    
    public SparseGradient copySign(final SparseGradient sign) {
        final long m = Double.doubleToLongBits(value);
        final long s = Double.doubleToLongBits(sign.value);
        if ((m >= 0 && s >= 0) || (m < 0 && s < 0)) { // Sign is currently OK
            return this;
        }
        return negate(); // flip sign
    }

    
    public SparseGradient copySign(final double sign) {
        final long m = Double.doubleToLongBits(value);
        final long s = Double.doubleToLongBits(sign);
        if ((m >= 0 && s >= 0) || (m < 0 && s < 0)) { // Sign is currently OK
            return this;
        }
        return negate(); // flip sign
    }

    
    public SparseGradient scalb(final int n) {
        final SparseGradient out = new SparseGradient(FastMath.scalb(value, n), Collections.<Integer, Double> emptyMap());
        for (Map.Entry<Integer, Double> entry : derivatives.entrySet()) {
            out.derivatives.put(entry.getKey(), FastMath.scalb(entry.getValue(), n));
        }
        return out;
    }

    
    public SparseGradient hypot(final SparseGradient y) {
        if (Double.isInfinite(value) || Double.isInfinite(y.value)) {
            return createConstant(Double.POSITIVE_INFINITY);
        } else if (Double.isNaN(value) || Double.isNaN(y.value)) {
            return createConstant(Double.NaN);
        } else {

            final int expX = FastMath.getExponent(value);
            final int expY = FastMath.getExponent(y.value);
            if (expX > expY + 27) {
                // y is negligible with respect to x
                return abs();
            } else if (expY > expX + 27) {
                // x is negligible with respect to y
                return y.abs();
            } else {

                // find an intermediate scale to avoid both overflow and underflow
                final int middleExp = (expX + expY) / 2;

                // scale parameters without losing precision
                final SparseGradient scaledX = scalb(-middleExp);
                final SparseGradient scaledY = y.scalb(-middleExp);

                // compute scaled hypotenuse
                final SparseGradient scaledH =
                        scaledX.multiply(scaledX).add(scaledY.multiply(scaledY)).sqrt();

                // remove scaling
                return scaledH.scalb(middleExp);

            }

        }
    }

    
    public static SparseGradient hypot(final SparseGradient x, final SparseGradient y) {
        return x.hypot(y);
    }

    
    public SparseGradient reciprocal() {
        return new SparseGradient(1.0 / value, -1.0 / (value * value), derivatives);
    }

    
    public SparseGradient sqrt() {
        final double sqrt = FastMath.sqrt(value);
        return new SparseGradient(sqrt, 0.5 / sqrt, derivatives);
    }

    
    public SparseGradient cbrt() {
        final double cbrt = FastMath.cbrt(value);
        return new SparseGradient(cbrt, 1.0 / (3 * cbrt * cbrt), derivatives);
    }

    
    public SparseGradient rootN(final int n) {
        if (n == 2) {
            return sqrt();
        } else if (n == 3) {
            return cbrt();
        } else {
            final double root = FastMath.pow(value, 1.0 / n);
            return new SparseGradient(root, 1.0 / (n * FastMath.pow(root, n - 1)), derivatives);
        }
    }

    
    public SparseGradient pow(final double p) {
        return new SparseGradient(FastMath.pow(value,  p), p * FastMath.pow(value,  p - 1), derivatives);
    }

    
    public SparseGradient pow(final int n) {
        if (n == 0) {
            return getField().getOne();
        } else {
            final double valueNm1 = FastMath.pow(value,  n - 1);
            return new SparseGradient(value * valueNm1, n * valueNm1, derivatives);
        }
    }

    
    public SparseGradient pow(final SparseGradient e) {
        return log().multiply(e).exp();
    }

    
    public static SparseGradient pow(final double a, final SparseGradient x) {
        if (a == 0) {
            if (x.value == 0) {
                return x.compose(1.0, Double.NEGATIVE_INFINITY);
            } else if (x.value < 0) {
                return x.compose(Double.NaN, Double.NaN);
            } else {
                return x.getField().getZero();
            }
        } else {
            final double ax = FastMath.pow(a, x.value);
            return new SparseGradient(ax, ax * FastMath.log(a), x.derivatives);
        }
    }

    
    public SparseGradient exp() {
        final double e = FastMath.exp(value);
        return new SparseGradient(e, e, derivatives);
    }

    
    public SparseGradient expm1() {
        return new SparseGradient(FastMath.expm1(value), FastMath.exp(value), derivatives);
    }

    
    public SparseGradient log() {
        return new SparseGradient(FastMath.log(value), 1.0 / value, derivatives);
    }

    
    public SparseGradient log10() {
        return new SparseGradient(FastMath.log10(value), 1.0 / (FastMath.log(10.0) * value), derivatives);
    }

    
    public SparseGradient log1p() {
        return new SparseGradient(FastMath.log1p(value), 1.0 / (1.0 + value), derivatives);
    }

    
    public SparseGradient cos() {
        return new SparseGradient(FastMath.cos(value), -FastMath.sin(value), derivatives);
    }

    
    public SparseGradient sin() {
        return new SparseGradient(FastMath.sin(value), FastMath.cos(value), derivatives);
    }

    
    public SparseGradient tan() {
        final double t = FastMath.tan(value);
        return new SparseGradient(t, 1 + t * t, derivatives);
    }

    
    public SparseGradient acos() {
        return new SparseGradient(FastMath.acos(value), -1.0 / FastMath.sqrt(1 - value * value), derivatives);
    }

    
    public SparseGradient asin() {
        return new SparseGradient(FastMath.asin(value), 1.0 / FastMath.sqrt(1 - value * value), derivatives);
    }

    
    public SparseGradient atan() {
        return new SparseGradient(FastMath.atan(value), 1.0 / (1 + value * value), derivatives);
    }

    
    public SparseGradient atan2(final SparseGradient x) {

        // compute r = sqrt(x^2+y^2)
        final SparseGradient r = multiply(this).add(x.multiply(x)).sqrt();

        final SparseGradient a;
        if (x.value >= 0) {

            // compute atan2(y, x) = 2 atan(y / (r + x))
            a = divide(r.add(x)).atan().multiply(2);

        } else {

            // compute atan2(y, x) = +/- pi - 2 atan(y / (r - x))
            final SparseGradient tmp = divide(r.subtract(x)).atan().multiply(-2);
            a = tmp.add(tmp.value <= 0 ? -FastMath.PI : FastMath.PI);

        }

        // fix value to take special cases (+0/+0, +0/-0, -0/+0, -0/-0, +/-infinity) correctly
        a.value = FastMath.atan2(value, x.value);

        return a;

    }

    
    public static SparseGradient atan2(final SparseGradient y, final SparseGradient x) {
        return y.atan2(x);
    }

    
    public SparseGradient cosh() {
        return new SparseGradient(FastMath.cosh(value), FastMath.sinh(value), derivatives);
    }

    
    public SparseGradient sinh() {
        return new SparseGradient(FastMath.sinh(value), FastMath.cosh(value), derivatives);
    }

    
    public SparseGradient tanh() {
        final double t = FastMath.tanh(value);
        return new SparseGradient(t, 1 - t * t, derivatives);
    }

    
    public SparseGradient acosh() {
        return new SparseGradient(FastMath.acosh(value), 1.0 / FastMath.sqrt(value * value - 1.0), derivatives);
    }

    
    public SparseGradient asinh() {
        return new SparseGradient(FastMath.asinh(value), 1.0 / FastMath.sqrt(value * value + 1.0), derivatives);
    }

    
    public SparseGradient atanh() {
        return new SparseGradient(FastMath.atanh(value), 1.0 / (1.0 - value * value), derivatives);
    }

    
    public SparseGradient toDegrees() {
        return new SparseGradient(FastMath.toDegrees(value), FastMath.toDegrees(1.0), derivatives);
    }

    
    public SparseGradient toRadians() {
        return new SparseGradient(FastMath.toRadians(value), FastMath.toRadians(1.0), derivatives);
    }

    
    public double taylor(final double ... delta) {
        double y = value;
        for (int i = 0; i < delta.length; ++i) {
            y += delta[i] * getDerivative(i);
        }
        return y;
    }

    
    public SparseGradient compose(final double f0, final double f1) {
        return new SparseGradient(f0, f1, derivatives);
    }

    
    public SparseGradient linearCombination(final SparseGradient[] a,
                                              final SparseGradient[] b)
        throws DimensionMismatchException {

        // compute a simple value, with all partial derivatives
        SparseGradient out = a[0].getField().getZero();
        for (int i = 0; i < a.length; ++i) {
            out = out.add(a[i].multiply(b[i]));
        }

        // recompute an accurate value, taking care of cancellations
        final double[] aDouble = new double[a.length];
        for (int i = 0; i < a.length; ++i) {
            aDouble[i] = a[i].getValue();
        }
        final double[] bDouble = new double[b.length];
        for (int i = 0; i < b.length; ++i) {
            bDouble[i] = b[i].getValue();
        }
        out.value = MathArrays.linearCombination(aDouble, bDouble);

        return out;

    }

    
    public SparseGradient linearCombination(final double[] a, final SparseGradient[] b) {

        // compute a simple value, with all partial derivatives
        SparseGradient out = b[0].getField().getZero();
        for (int i = 0; i < a.length; ++i) {
            out = out.add(b[i].multiply(a[i]));
        }

        // recompute an accurate value, taking care of cancellations
        final double[] bDouble = new double[b.length];
        for (int i = 0; i < b.length; ++i) {
            bDouble[i] = b[i].getValue();
        }
        out.value = MathArrays.linearCombination(a, bDouble);

        return out;

    }

    
    public SparseGradient linearCombination(final SparseGradient a1, final SparseGradient b1,
                                              final SparseGradient a2, final SparseGradient b2) {

        // compute a simple value, with all partial derivatives
        SparseGradient out = a1.multiply(b1).add(a2.multiply(b2));

        // recompute an accurate value, taking care of cancellations
        out.value = MathArrays.linearCombination(a1.value, b1.value, a2.value, b2.value);

        return out;

    }

    
    public SparseGradient linearCombination(final double a1, final SparseGradient b1,
                                              final double a2, final SparseGradient b2) {

        // compute a simple value, with all partial derivatives
        SparseGradient out = b1.multiply(a1).add(b2.multiply(a2));

        // recompute an accurate value, taking care of cancellations
        out.value = MathArrays.linearCombination(a1, b1.value, a2, b2.value);

        return out;

    }

    
    public SparseGradient linearCombination(final SparseGradient a1, final SparseGradient b1,
                                              final SparseGradient a2, final SparseGradient b2,
                                              final SparseGradient a3, final SparseGradient b3) {

        // compute a simple value, with all partial derivatives
        SparseGradient out = a1.multiply(b1).add(a2.multiply(b2)).add(a3.multiply(b3));

        // recompute an accurate value, taking care of cancellations
        out.value = MathArrays.linearCombination(a1.value, b1.value,
                                                 a2.value, b2.value,
                                                 a3.value, b3.value);

        return out;

    }

    
    public SparseGradient linearCombination(final double a1, final SparseGradient b1,
                                              final double a2, final SparseGradient b2,
                                              final double a3, final SparseGradient b3) {

        // compute a simple value, with all partial derivatives
        SparseGradient out = b1.multiply(a1).add(b2.multiply(a2)).add(b3.multiply(a3));

        // recompute an accurate value, taking care of cancellations
        out.value = MathArrays.linearCombination(a1, b1.value,
                                                 a2, b2.value,
                                                 a3, b3.value);

        return out;

    }

    
    public SparseGradient linearCombination(final SparseGradient a1, final SparseGradient b1,
                                              final SparseGradient a2, final SparseGradient b2,
                                              final SparseGradient a3, final SparseGradient b3,
                                              final SparseGradient a4, final SparseGradient b4) {

        // compute a simple value, with all partial derivatives
        SparseGradient out = a1.multiply(b1).add(a2.multiply(b2)).add(a3.multiply(b3)).add(a4.multiply(b4));

        // recompute an accurate value, taking care of cancellations
        out.value = MathArrays.linearCombination(a1.value, b1.value,
                                                 a2.value, b2.value,
                                                 a3.value, b3.value,
                                                 a4.value, b4.value);

        return out;

    }

    
    public SparseGradient linearCombination(final double a1, final SparseGradient b1,
                                              final double a2, final SparseGradient b2,
                                              final double a3, final SparseGradient b3,
                                              final double a4, final SparseGradient b4) {

        // compute a simple value, with all partial derivatives
        SparseGradient out = b1.multiply(a1).add(b2.multiply(a2)).add(b3.multiply(a3)).add(b4.multiply(a4));

        // recompute an accurate value, taking care of cancellations
        out.value = MathArrays.linearCombination(a1, b1.value,
                                                 a2, b2.value,
                                                 a3, b3.value,
                                                 a4, b4.value);

        return out;

    }

    
    @Override
    public boolean equals(Object other) {

        if (this == other) {
            return true;
        }

        if (other instanceof SparseGradient) {
            final SparseGradient rhs = (SparseGradient)other;
            if (!Precision.equals(value, rhs.value, 1)) {
                return false;
            }
            if (derivatives.size() != rhs.derivatives.size()) {
                return false;
            }
            for (final Map.Entry<Integer, Double> entry : derivatives.entrySet()) {
                if (!rhs.derivatives.containsKey(entry.getKey())) {
                    return false;
                }
                if (!Precision.equals(entry.getValue(), rhs.derivatives.get(entry.getKey()), 1)) {
                    return false;
                }
            }
            return true;
        }

        return false;

    }

    
    @Override
    public int hashCode() {
        return 743 + 809 * MathUtils.hash(value) + 167 * derivatives.hashCode();
    }

}
