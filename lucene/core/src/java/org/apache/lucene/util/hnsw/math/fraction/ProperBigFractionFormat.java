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
package org.apache.lucene.util.hnsw.math.fraction;

import java.math.BigInteger;
import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;

import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;


public class ProperBigFractionFormat extends BigFractionFormat {

    
    private static final long serialVersionUID = -6337346779577272307L;

    
    private NumberFormat wholeFormat;

    
    public ProperBigFractionFormat() {
        this(getDefaultNumberFormat());
    }

    
    public ProperBigFractionFormat(final NumberFormat format) {
        this(format, (NumberFormat)format.clone(), (NumberFormat)format.clone());
    }

    
    public ProperBigFractionFormat(final NumberFormat wholeFormat,
                                   final NumberFormat numeratorFormat,
                                   final NumberFormat denominatorFormat) {
        super(numeratorFormat, denominatorFormat);
        setWholeFormat(wholeFormat);
    }

    
    @Override
    public StringBuffer format(final BigFraction fraction,
                               final StringBuffer toAppendTo, final FieldPosition pos) {

        pos.setBeginIndex(0);
        pos.setEndIndex(0);

        BigInteger num = fraction.getNumerator();
        BigInteger den = fraction.getDenominator();
        BigInteger whole = num.divide(den);
        num = num.remainder(den);

        if (!BigInteger.ZERO.equals(whole)) {
            getWholeFormat().format(whole, toAppendTo, pos);
            toAppendTo.append(' ');
            if (num.compareTo(BigInteger.ZERO) < 0) {
                num = num.negate();
            }
        }
        getNumeratorFormat().format(num, toAppendTo, pos);
        toAppendTo.append(" / ");
        getDenominatorFormat().format(den, toAppendTo, pos);

        return toAppendTo;
    }

    
    public NumberFormat getWholeFormat() {
        return wholeFormat;
    }

    
    @Override
    public BigFraction parse(final String source, final ParsePosition pos) {
        // try to parse improper BigFraction
        BigFraction ret = super.parse(source, pos);
        if (ret != null) {
            return ret;
        }

        final int initialIndex = pos.getIndex();

        // parse whitespace
        parseAndIgnoreWhitespace(source, pos);

        // parse whole
        BigInteger whole = parseNextBigInteger(source, pos);
        if (whole == null) {
            // invalid integer number
            // set index back to initial, error index should already be set
            // character examined.
            pos.setIndex(initialIndex);
            return null;
        }

        // parse whitespace
        parseAndIgnoreWhitespace(source, pos);

        // parse numerator
        BigInteger num = parseNextBigInteger(source, pos);
        if (num == null) {
            // invalid integer number
            // set index back to initial, error index should already be set
            // character examined.
            pos.setIndex(initialIndex);
            return null;
        }

        if (num.compareTo(BigInteger.ZERO) < 0) {
            // minus signs should be leading, invalid expression
            pos.setIndex(initialIndex);
            return null;
        }

        // parse '/'
        final int startIndex = pos.getIndex();
        final char c = parseNextCharacter(source, pos);
        switch (c) {
        case 0 :
            // no '/'
            // return num as a BigFraction
            return new BigFraction(num);
        case '/' :
            // found '/', continue parsing denominator
            break;
        default :
            // invalid '/'
            // set index back to initial, error index should be the last
            // character examined.
            pos.setIndex(initialIndex);
            pos.setErrorIndex(startIndex);
            return null;
        }

        // parse whitespace
        parseAndIgnoreWhitespace(source, pos);

        // parse denominator
        final BigInteger den = parseNextBigInteger(source, pos);
        if (den == null) {
            // invalid integer number
            // set index back to initial, error index should already be set
            // character examined.
            pos.setIndex(initialIndex);
            return null;
        }

        if (den.compareTo(BigInteger.ZERO) < 0) {
            // minus signs must be leading, invalid
            pos.setIndex(initialIndex);
            return null;
        }

        boolean wholeIsNeg = whole.compareTo(BigInteger.ZERO) < 0;
        if (wholeIsNeg) {
            whole = whole.negate();
        }
        num = whole.multiply(den).add(num);
        if (wholeIsNeg) {
            num = num.negate();
        }

        return new BigFraction(num, den);

    }

    
    public void setWholeFormat(final NumberFormat format) {
        if (format == null) {
            throw new NullArgumentException(LocalizedFormats.WHOLE_FORMAT);
        }
        this.wholeFormat = format;
    }
}
