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

import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.Locale;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathParseException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class FractionFormat extends AbstractFormat {

    
    private static final long serialVersionUID = 3008655719530972611L;

    
    public FractionFormat() {
    }

    
    public FractionFormat(final NumberFormat format) {
        super(format);
    }

    
    public FractionFormat(final NumberFormat numeratorFormat,
                          final NumberFormat denominatorFormat) {
        super(numeratorFormat, denominatorFormat);
    }

    
    public static Locale[] getAvailableLocales() {
        return NumberFormat.getAvailableLocales();
    }

    
    public static String formatFraction(Fraction f) {
        return getImproperInstance().format(f);
    }

    
    public static FractionFormat getImproperInstance() {
        return getImproperInstance(Locale.getDefault());
    }

    
    public static FractionFormat getImproperInstance(final Locale locale) {
        return new FractionFormat(getDefaultNumberFormat(locale));
    }

    
    public static FractionFormat getProperInstance() {
        return getProperInstance(Locale.getDefault());
    }

    
    public static FractionFormat getProperInstance(final Locale locale) {
        return new ProperFractionFormat(getDefaultNumberFormat(locale));
    }

    
    protected static NumberFormat getDefaultNumberFormat() {
        return getDefaultNumberFormat(Locale.getDefault());
    }

    
    public StringBuffer format(final Fraction fraction,
                               final StringBuffer toAppendTo, final FieldPosition pos) {

        pos.setBeginIndex(0);
        pos.setEndIndex(0);

        getNumeratorFormat().format(fraction.getNumerator(), toAppendTo, pos);
        toAppendTo.append(" / ");
        getDenominatorFormat().format(fraction.getDenominator(), toAppendTo,
            pos);

        return toAppendTo;
    }

    
    @Override
    public StringBuffer format(final Object obj,
                               final StringBuffer toAppendTo, final FieldPosition pos)
        throws FractionConversionException, MathIllegalArgumentException {
        StringBuffer ret = null;

        if (obj instanceof Fraction) {
            ret = format((Fraction) obj, toAppendTo, pos);
        } else if (obj instanceof Number) {
            ret = format(new Fraction(((Number) obj).doubleValue()), toAppendTo, pos);
        } else {
            throw new MathIllegalArgumentException(LocalizedFormats.CANNOT_FORMAT_OBJECT_TO_FRACTION);
        }

        return ret;
    }

    
    @Override
    public Fraction parse(final String source) throws MathParseException {
        final ParsePosition parsePosition = new ParsePosition(0);
        final Fraction result = parse(source, parsePosition);
        if (parsePosition.getIndex() == 0) {
            throw new MathParseException(source, parsePosition.getErrorIndex(), Fraction.class);
        }
        return result;
    }

    
    @Override
    public Fraction parse(final String source, final ParsePosition pos) {
        final int initialIndex = pos.getIndex();

        // parse whitespace
        parseAndIgnoreWhitespace(source, pos);

        // parse numerator
        final Number num = getNumeratorFormat().parse(source, pos);
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
            // return num as a fraction
            return new Fraction(num.intValue(), 1);
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
        final Number den = getDenominatorFormat().parse(source, pos);
        if (den == null) {
            // invalid integer number
            // set index back to initial, error index should already be set
            // character examined.
            pos.setIndex(initialIndex);
            return null;
        }

        return new Fraction(num.intValue(), den.intValue());
    }

}
