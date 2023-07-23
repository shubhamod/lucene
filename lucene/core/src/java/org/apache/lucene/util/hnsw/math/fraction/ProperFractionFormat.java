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

import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class ProperFractionFormat extends FractionFormat {

    
    private static final long serialVersionUID = 760934726031766749L;

    
    private NumberFormat wholeFormat;

    
    public ProperFractionFormat() {
        this(getDefaultNumberFormat());
    }

    
    public ProperFractionFormat(NumberFormat format) {
        this(format, (NumberFormat)format.clone(), (NumberFormat)format.clone());
    }

    
    public ProperFractionFormat(NumberFormat wholeFormat,
            NumberFormat numeratorFormat,
            NumberFormat denominatorFormat)
    {
        super(numeratorFormat, denominatorFormat);
        setWholeFormat(wholeFormat);
    }

    
    @Override
    public StringBuffer format(Fraction fraction, StringBuffer toAppendTo,
            FieldPosition pos) {

        pos.setBeginIndex(0);
        pos.setEndIndex(0);

        int num = fraction.getNumerator();
        int den = fraction.getDenominator();
        int whole = num / den;
        num %= den;

        if (whole != 0) {
            getWholeFormat().format(whole, toAppendTo, pos);
            toAppendTo.append(' ');
            num = FastMath.abs(num);
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
    public Fraction parse(String source, ParsePosition pos) {
        // try to parse improper fraction
        Fraction ret = super.parse(source, pos);
        if (ret != null) {
            return ret;
        }

        int initialIndex = pos.getIndex();

        // parse whitespace
        parseAndIgnoreWhitespace(source, pos);

        // parse whole
        Number whole = getWholeFormat().parse(source, pos);
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
        Number num = getNumeratorFormat().parse(source, pos);
        if (num == null) {
            // invalid integer number
            // set index back to initial, error index should already be set
            // character examined.
            pos.setIndex(initialIndex);
            return null;
        }

        if (num.intValue() < 0) {
            // minus signs should be leading, invalid expression
            pos.setIndex(initialIndex);
            return null;
        }

        // parse '/'
        int startIndex = pos.getIndex();
        char c = parseNextCharacter(source, pos);
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
        Number den = getDenominatorFormat().parse(source, pos);
        if (den == null) {
            // invalid integer number
            // set index back to initial, error index should already be set
            // character examined.
            pos.setIndex(initialIndex);
            return null;
        }

        if (den.intValue() < 0) {
            // minus signs must be leading, invalid
            pos.setIndex(initialIndex);
            return null;
        }

        int w = whole.intValue();
        int n = num.intValue();
        int d = den.intValue();
        return new Fraction(((FastMath.abs(w) * d) + n) * MathUtils.copySign(1, w), d);
    }

    
    public void setWholeFormat(NumberFormat format) {
        if (format == null) {
            throw new NullArgumentException(LocalizedFormats.WHOLE_FORMAT);
        }
        this.wholeFormat = format;
    }
}
