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

import java.io.Serializable;
import java.math.BigInteger;
import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.Locale;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathParseException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class BigFractionFormat extends AbstractFormat implements Serializable {

    
    private static final long serialVersionUID = -2932167925527338976L;

    
    public BigFractionFormat() {
    }

    
    public BigFractionFormat(final NumberFormat format) {
        super(format);
    }

    
    public BigFractionFormat(final NumberFormat numeratorFormat,
                             final NumberFormat denominatorFormat) {
        super(numeratorFormat, denominatorFormat);
    }

    
    public static Locale[] getAvailableLocales() {
        return NumberFormat.getAvailableLocales();
    }

    
    public static String formatBigFraction(final BigFraction f) {
        return getImproperInstance().format(f);
    }

    
    public static BigFractionFormat getImproperInstance() {
        return getImproperInstance(Locale.getDefault());
    }

    
    public static BigFractionFormat getImproperInstance(final Locale locale) {
        return new BigFractionFormat(getDefaultNumberFormat(locale));
    }

    
    public static BigFractionFormat getProperInstance() {
        return getProperInstance(Locale.getDefault());
    }

    
    public static BigFractionFormat getProperInstance(final Locale locale) {
        return new ProperBigFractionFormat(getDefaultNumberFormat(locale));
    }

    
    public StringBuffer format(final BigFraction BigFraction,
                               final StringBuffer toAppendTo, final FieldPosition pos) {

        pos.setBeginIndex(0);
        pos.setEndIndex(0);

        getNumeratorFormat().format(BigFraction.getNumerator(), toAppendTo, pos);
        toAppendTo.append(" / ");
        getDenominatorFormat().format(BigFraction.getDenominator(), toAppendTo, pos);

        return toAppendTo;
    }

    
    @Override
    public StringBuffer format(final Object obj,
                               final StringBuffer toAppendTo, final FieldPosition pos) {

        final StringBuffer ret;
        if (obj instanceof BigFraction) {
            ret = format((BigFraction) obj, toAppendTo, pos);
        } else if (obj instanceof BigInteger) {
            ret = format(new BigFraction((BigInteger) obj), toAppendTo, pos);
        } else if (obj instanceof Number) {
            ret = format(new BigFraction(((Number) obj).doubleValue()),
                         toAppendTo, pos);
        } else {
            throw new MathIllegalArgumentException(LocalizedFormats.CANNOT_FORMAT_OBJECT_TO_FRACTION);
        }

        return ret;
    }

    
    @Override
    public BigFraction parse(final String source) throws MathParseException {
        final ParsePosition parsePosition = new ParsePosition(0);
        final BigFraction result = parse(source, parsePosition);
        if (parsePosition.getIndex() == 0) {
            throw new MathParseException(source, parsePosition.getErrorIndex(), BigFraction.class);
        }
        return result;
    }

    
    @Override
    public BigFraction parse(final String source, final ParsePosition pos) {
        final int initialIndex = pos.getIndex();

        // parse whitespace
        parseAndIgnoreWhitespace(source, pos);

        // parse numerator
        final BigInteger num = parseNextBigInteger(source, pos);
        if (num == null) {
            // invalid integer number
            // set index back to initial, error index should already be set
            // character examined.
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

        return new BigFraction(num, den);
    }

    
    protected BigInteger parseNextBigInteger(final String source,
                                             final ParsePosition pos) {

        final int start = pos.getIndex();
         int end = (source.charAt(start) == '-') ? (start + 1) : start;
         while((end < source.length()) &&
               Character.isDigit(source.charAt(end))) {
             ++end;
         }

         try {
             BigInteger n = new BigInteger(source.substring(start, end));
             pos.setIndex(end);
             return n;
         } catch (NumberFormatException nfe) {
             pos.setErrorIndex(start);
             return null;
         }

    }

}
