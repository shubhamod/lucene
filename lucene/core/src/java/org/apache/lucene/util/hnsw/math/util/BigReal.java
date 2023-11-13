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
package org.apache.lucene.util.hnsw.math.util;


import java.io.Serializable;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class BigReal implements FieldElement<BigReal>, Comparable<BigReal>, Serializable {

    
    public static final BigReal ZERO = new BigReal(BigDecimal.ZERO);

    
    public static final BigReal ONE = new BigReal(BigDecimal.ONE);

    
    private static final long serialVersionUID = 4984534880991310382L;

    
    private final BigDecimal d;

    
    private RoundingMode roundingMode = RoundingMode.HALF_UP;

    
    private int scale = 64;

    
    public BigReal(BigDecimal val) {
        d =  val;
    }

    
    public BigReal(BigInteger val) {
        d = new BigDecimal(val);
    }

    
    public BigReal(BigInteger unscaledVal, int scale) {
        d = new BigDecimal(unscaledVal, scale);
    }

    
    public BigReal(BigInteger unscaledVal, int scale, MathContext mc) {
        d = new BigDecimal(unscaledVal, scale, mc);
    }

    
    public BigReal(BigInteger val, MathContext mc) {
        d = new BigDecimal(val, mc);
    }

    
    public BigReal(char[] in) {
        d = new BigDecimal(in);
    }

    
    public BigReal(char[] in, int offset, int len) {
        d = new BigDecimal(in, offset, len);
    }

    
    public BigReal(char[] in, int offset, int len, MathContext mc) {
        d = new BigDecimal(in, offset, len, mc);
    }

    
    public BigReal(char[] in, MathContext mc) {
        d = new BigDecimal(in, mc);
    }

    
    public BigReal(double val) {
        d = new BigDecimal(val);
    }

    
    public BigReal(double val, MathContext mc) {
        d = new BigDecimal(val, mc);
    }

    
    public BigReal(int val) {
        d = new BigDecimal(val);
    }

    
    public BigReal(int val, MathContext mc) {
        d = new BigDecimal(val, mc);
    }

    
    public BigReal(long val) {
        d = new BigDecimal(val);
    }

    
    public BigReal(long val, MathContext mc) {
        d = new BigDecimal(val, mc);
    }

    
    public BigReal(String val) {
        d = new BigDecimal(val);
    }

    
    public BigReal(String val, MathContext mc)  {
        d = new BigDecimal(val, mc);
    }

    
    public RoundingMode getRoundingMode() {
        return roundingMode;
    }

    
    public void setRoundingMode(RoundingMode roundingMode) {
        this.roundingMode = roundingMode;
    }

    
    public int getScale() {
        return scale;
    }

    
    public void setScale(int scale) {
        this.scale = scale;
    }

    
    public BigReal add(BigReal a) {
        return new BigReal(d.add(a.d));
    }

    
    public BigReal subtract(BigReal a) {
        return new BigReal(d.subtract(a.d));
    }

    
    public BigReal negate() {
        return new BigReal(d.negate());
    }

    
    public BigReal divide(BigReal a) throws MathArithmeticException {
        try {
            return new BigReal(d.divide(a.d, scale, roundingMode));
        } catch (ArithmeticException e) {
            // Division by zero has occurred
            throw new MathArithmeticException(LocalizedFormats.ZERO_NOT_ALLOWED);
        }
    }

    
    public BigReal reciprocal() throws MathArithmeticException {
        try {
            return new BigReal(BigDecimal.ONE.divide(d, scale, roundingMode));
        } catch (ArithmeticException e) {
            // Division by zero has occurred
            throw new MathArithmeticException(LocalizedFormats.ZERO_NOT_ALLOWED);
        }
    }

    
    public BigReal multiply(BigReal a) {
        return new BigReal(d.multiply(a.d));
    }

    
    public BigReal multiply(final int n) {
        return new BigReal(d.multiply(new BigDecimal(n)));
    }

    
    public int compareTo(BigReal a) {
        return d.compareTo(a.d);
    }

    
    public double doubleValue() {
        return d.doubleValue();
    }

    
    public BigDecimal bigDecimalValue() {
        return d;
    }

    
    @Override
    public boolean equals(Object other) {
        if (this == other){
            return true;
        }

        if (other instanceof BigReal){
            return d.equals(((BigReal) other).d);
        }
        return false;
    }

    
    @Override
    public int hashCode() {
        return d.hashCode();
    }

    
    public Field<BigReal> getField() {
        return BigRealField.getInstance();
    }
}
