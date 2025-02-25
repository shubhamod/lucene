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

package org.apache.lucene.util.hnsw.math.complex;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class Complex implements FieldElement<Complex>, Serializable  {
    
    public static final Complex I = new Complex(0.0, 1.0);
    // CHECKSTYLE: stop ConstantName
    
    public static final Complex NaN = new Complex(Double.NaN, Double.NaN);
    // CHECKSTYLE: resume ConstantName
    
    public static final Complex INF = new Complex(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
    
    public static final Complex ONE = new Complex(1.0, 0.0);
    
    public static final Complex ZERO = new Complex(0.0, 0.0);

    
    private static final long serialVersionUID = -6195664516687396620L;

    
    private final double imaginary;
    
    private final double real;
    
    private final transient boolean isNaN;
    
    private final transient boolean isInfinite;

    
    public Complex(double real) {
        this(real, 0.0);
    }

    
    public Complex(double real, double imaginary) {
        this.real = real;
        this.imaginary = imaginary;

        isNaN = Double.isNaN(real) || Double.isNaN(imaginary);
        isInfinite = !isNaN &&
            (Double.isInfinite(real) || Double.isInfinite(imaginary));
    }

    
    public double abs() {
        if (isNaN) {
            return Double.NaN;
        }
        if (isInfinite()) {
            return Double.POSITIVE_INFINITY;
        }
        if (FastMath.abs(real) < FastMath.abs(imaginary)) {
            if (imaginary == 0.0) {
                return FastMath.abs(real);
            }
            double q = real / imaginary;
            return FastMath.abs(imaginary) * FastMath.sqrt(1 + q * q);
        } else {
            if (real == 0.0) {
                return FastMath.abs(imaginary);
            }
            double q = imaginary / real;
            return FastMath.abs(real) * FastMath.sqrt(1 + q * q);
        }
    }

    
    public Complex add(Complex addend) throws NullArgumentException {
        MathUtils.checkNotNull(addend);
        if (isNaN || addend.isNaN) {
            return NaN;
        }

        return createComplex(real + addend.getReal(),
                             imaginary + addend.getImaginary());
    }

    
    public Complex add(double addend) {
        if (isNaN || Double.isNaN(addend)) {
            return NaN;
        }

        return createComplex(real + addend, imaginary);
    }

     
    public Complex conjugate() {
        if (isNaN) {
            return NaN;
        }

        return createComplex(real, -imaginary);
    }

    
    public Complex divide(Complex divisor)
        throws NullArgumentException {
        MathUtils.checkNotNull(divisor);
        if (isNaN || divisor.isNaN) {
            return NaN;
        }

        final double c = divisor.getReal();
        final double d = divisor.getImaginary();
        if (c == 0.0 && d == 0.0) {
            return NaN;
        }

        if (divisor.isInfinite() && !isInfinite()) {
            return ZERO;
        }

        if (FastMath.abs(c) < FastMath.abs(d)) {
            double q = c / d;
            double denominator = c * q + d;
            return createComplex((real * q + imaginary) / denominator,
                (imaginary * q - real) / denominator);
        } else {
            double q = d / c;
            double denominator = d * q + c;
            return createComplex((imaginary * q + real) / denominator,
                (imaginary - real * q) / denominator);
        }
    }

    
    public Complex divide(double divisor) {
        if (isNaN || Double.isNaN(divisor)) {
            return NaN;
        }
        if (divisor == 0d) {
            return NaN;
        }
        if (Double.isInfinite(divisor)) {
            return !isInfinite() ? ZERO : NaN;
        }
        return createComplex(real / divisor,
                             imaginary  / divisor);
    }

    
    public Complex reciprocal() {
        if (isNaN) {
            return NaN;
        }

        if (real == 0.0 && imaginary == 0.0) {
            return INF;
        }

        if (isInfinite) {
            return ZERO;
        }

        if (FastMath.abs(real) < FastMath.abs(imaginary)) {
            double q = real / imaginary;
            double scale = 1. / (real * q + imaginary);
            return createComplex(scale * q, -scale);
        } else {
            double q = imaginary / real;
            double scale = 1. / (imaginary * q + real);
            return createComplex(scale, -scale * q);
        }
    }

    
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other instanceof Complex){
            Complex c = (Complex) other;
            if (c.isNaN) {
                return isNaN;
            } else {
                return MathUtils.equals(real, c.real) &&
                    MathUtils.equals(imaginary, c.imaginary);
            }
        }
        return false;
    }

    
    public static boolean equals(Complex x, Complex y, int maxUlps) {
        return Precision.equals(x.real, y.real, maxUlps) &&
            Precision.equals(x.imaginary, y.imaginary, maxUlps);
    }

    
    public static boolean equals(Complex x, Complex y) {
        return equals(x, y, 1);
    }

    
    public static boolean equals(Complex x, Complex y, double eps) {
        return Precision.equals(x.real, y.real, eps) &&
            Precision.equals(x.imaginary, y.imaginary, eps);
    }

    
    public static boolean equalsWithRelativeTolerance(Complex x, Complex y,
                                                      double eps) {
        return Precision.equalsWithRelativeTolerance(x.real, y.real, eps) &&
            Precision.equalsWithRelativeTolerance(x.imaginary, y.imaginary, eps);
    }

    
    @Override
    public int hashCode() {
        if (isNaN) {
            return 7;
        }
        return 37 * (17 * MathUtils.hash(imaginary) +
            MathUtils.hash(real));
    }

    
    public double getImaginary() {
        return imaginary;
    }

    
    public double getReal() {
        return real;
    }

    
    public boolean isNaN() {
        return isNaN;
    }

    
    public boolean isInfinite() {
        return isInfinite;
    }

    
    public Complex multiply(Complex factor)
        throws NullArgumentException {
        MathUtils.checkNotNull(factor);
        if (isNaN || factor.isNaN) {
            return NaN;
        }
        if (Double.isInfinite(real) ||
            Double.isInfinite(imaginary) ||
            Double.isInfinite(factor.real) ||
            Double.isInfinite(factor.imaginary)) {
            // we don't use isInfinite() to avoid testing for NaN again
            return INF;
        }
        return createComplex(real * factor.real - imaginary * factor.imaginary,
                             real * factor.imaginary + imaginary * factor.real);
    }

    
    public Complex multiply(final int factor) {
        if (isNaN) {
            return NaN;
        }
        if (Double.isInfinite(real) ||
            Double.isInfinite(imaginary)) {
            return INF;
        }
        return createComplex(real * factor, imaginary * factor);
    }

    
    public Complex multiply(double factor) {
        if (isNaN || Double.isNaN(factor)) {
            return NaN;
        }
        if (Double.isInfinite(real) ||
            Double.isInfinite(imaginary) ||
            Double.isInfinite(factor)) {
            // we don't use isInfinite() to avoid testing for NaN again
            return INF;
        }
        return createComplex(real * factor, imaginary * factor);
    }

    
    public Complex negate() {
        if (isNaN) {
            return NaN;
        }

        return createComplex(-real, -imaginary);
    }

    
    public Complex subtract(Complex subtrahend)
        throws NullArgumentException {
        MathUtils.checkNotNull(subtrahend);
        if (isNaN || subtrahend.isNaN) {
            return NaN;
        }

        return createComplex(real - subtrahend.getReal(),
                             imaginary - subtrahend.getImaginary());
    }

    
    public Complex subtract(double subtrahend) {
        if (isNaN || Double.isNaN(subtrahend)) {
            return NaN;
        }
        return createComplex(real - subtrahend, imaginary);
    }

    
    public Complex acos() {
        if (isNaN) {
            return NaN;
        }

        return this.add(this.sqrt1z().multiply(I)).log().multiply(I.negate());
    }

    
    public Complex asin() {
        if (isNaN) {
            return NaN;
        }

        return sqrt1z().add(this.multiply(I)).log().multiply(I.negate());
    }

    
    public Complex atan() {
        if (isNaN) {
            return NaN;
        }

        return this.add(I).divide(I.subtract(this)).log()
                .multiply(I.divide(createComplex(2.0, 0.0)));
    }

    
    public Complex cos() {
        if (isNaN) {
            return NaN;
        }

        return createComplex(FastMath.cos(real) * FastMath.cosh(imaginary),
                             -FastMath.sin(real) * FastMath.sinh(imaginary));
    }

    
    public Complex cosh() {
        if (isNaN) {
            return NaN;
        }

        return createComplex(FastMath.cosh(real) * FastMath.cos(imaginary),
                             FastMath.sinh(real) * FastMath.sin(imaginary));
    }

    
    public Complex exp() {
        if (isNaN) {
            return NaN;
        }

        double expReal = FastMath.exp(real);
        return createComplex(expReal *  FastMath.cos(imaginary),
                             expReal * FastMath.sin(imaginary));
    }

    
    public Complex log() {
        if (isNaN) {
            return NaN;
        }

        return createComplex(FastMath.log(abs()),
                             FastMath.atan2(imaginary, real));
    }

    
    public Complex pow(Complex x)
        throws NullArgumentException {
        MathUtils.checkNotNull(x);
        return this.log().multiply(x).exp();
    }

    
     public Complex pow(double x) {
        return this.log().multiply(x).exp();
    }

    
    public Complex sin() {
        if (isNaN) {
            return NaN;
        }

        return createComplex(FastMath.sin(real) * FastMath.cosh(imaginary),
                             FastMath.cos(real) * FastMath.sinh(imaginary));
    }

    
    public Complex sinh() {
        if (isNaN) {
            return NaN;
        }

        return createComplex(FastMath.sinh(real) * FastMath.cos(imaginary),
            FastMath.cosh(real) * FastMath.sin(imaginary));
    }

    
    public Complex sqrt() {
        if (isNaN) {
            return NaN;
        }

        if (real == 0.0 && imaginary == 0.0) {
            return createComplex(0.0, 0.0);
        }

        double t = FastMath.sqrt((FastMath.abs(real) + abs()) / 2.0);
        if (real >= 0.0) {
            return createComplex(t, imaginary / (2.0 * t));
        } else {
            return createComplex(FastMath.abs(imaginary) / (2.0 * t),
                                 FastMath.copySign(1d, imaginary) * t);
        }
    }

    
    public Complex sqrt1z() {
        return createComplex(1.0, 0.0).subtract(this.multiply(this)).sqrt();
    }

    
    public Complex tan() {
        if (isNaN || Double.isInfinite(real)) {
            return NaN;
        }
        if (imaginary > 20.0) {
            return createComplex(0.0, 1.0);
        }
        if (imaginary < -20.0) {
            return createComplex(0.0, -1.0);
        }

        double real2 = 2.0 * real;
        double imaginary2 = 2.0 * imaginary;
        double d = FastMath.cos(real2) + FastMath.cosh(imaginary2);

        return createComplex(FastMath.sin(real2) / d,
                             FastMath.sinh(imaginary2) / d);
    }

    
    public Complex tanh() {
        if (isNaN || Double.isInfinite(imaginary)) {
            return NaN;
        }
        if (real > 20.0) {
            return createComplex(1.0, 0.0);
        }
        if (real < -20.0) {
            return createComplex(-1.0, 0.0);
        }
        double real2 = 2.0 * real;
        double imaginary2 = 2.0 * imaginary;
        double d = FastMath.cosh(real2) + FastMath.cos(imaginary2);

        return createComplex(FastMath.sinh(real2) / d,
                             FastMath.sin(imaginary2) / d);
    }



    
    public double getArgument() {
        return FastMath.atan2(getImaginary(), getReal());
    }

    
    public List<Complex> nthRoot(int n) throws NotPositiveException {

        if (n <= 0) {
            throw new NotPositiveException(LocalizedFormats.CANNOT_COMPUTE_NTH_ROOT_FOR_NEGATIVE_N,
                                           n);
        }

        final List<Complex> result = new ArrayList<Complex>();

        if (isNaN) {
            result.add(NaN);
            return result;
        }
        if (isInfinite()) {
            result.add(INF);
            return result;
        }

        // nth root of abs -- faster / more accurate to use a solver here?
        final double nthRootOfAbs = FastMath.pow(abs(), 1.0 / n);

        // Compute nth roots of complex number with k = 0, 1, ... n-1
        final double nthPhi = getArgument() / n;
        final double slice = 2 * FastMath.PI / n;
        double innerPart = nthPhi;
        for (int k = 0; k < n ; k++) {
            // inner part
            final double realPart = nthRootOfAbs *  FastMath.cos(innerPart);
            final double imaginaryPart = nthRootOfAbs *  FastMath.sin(innerPart);
            result.add(createComplex(realPart, imaginaryPart));
            innerPart += slice;
        }

        return result;
    }

    
    protected Complex createComplex(double realPart,
                                    double imaginaryPart) {
        return new Complex(realPart, imaginaryPart);
    }

    
    public static Complex valueOf(double realPart,
                                  double imaginaryPart) {
        if (Double.isNaN(realPart) ||
            Double.isNaN(imaginaryPart)) {
            return NaN;
        }
        return new Complex(realPart, imaginaryPart);
    }

    
    public static Complex valueOf(double realPart) {
        if (Double.isNaN(realPart)) {
            return NaN;
        }
        return new Complex(realPart);
    }

    
    protected final Object readResolve() {
        return createComplex(real, imaginary);
    }

    
    public ComplexField getField() {
        return ComplexField.getInstance();
    }

    
    @Override
    public String toString() {
        return "(" + real + ", " + imaginary + ")";
    }

}
