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

import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.Locale;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathParseException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.CompositeFormat;


public class ComplexFormat {

     
    private static final String DEFAULT_IMAGINARY_CHARACTER = "i";
    
    private final String imaginaryCharacter;
    
    private final NumberFormat imaginaryFormat;
    
    private final NumberFormat realFormat;

    
    public ComplexFormat() {
        this.imaginaryCharacter = DEFAULT_IMAGINARY_CHARACTER;
        this.imaginaryFormat = CompositeFormat.getDefaultNumberFormat();
        this.realFormat = imaginaryFormat;
    }

    
    public ComplexFormat(NumberFormat format) throws NullArgumentException {
        if (format == null) {
            throw new NullArgumentException(LocalizedFormats.IMAGINARY_FORMAT);
        }
        this.imaginaryCharacter = DEFAULT_IMAGINARY_CHARACTER;
        this.imaginaryFormat = format;
        this.realFormat = format;
    }

    
    public ComplexFormat(NumberFormat realFormat, NumberFormat imaginaryFormat)
        throws NullArgumentException {
        if (imaginaryFormat == null) {
            throw new NullArgumentException(LocalizedFormats.IMAGINARY_FORMAT);
        }
        if (realFormat == null) {
            throw new NullArgumentException(LocalizedFormats.REAL_FORMAT);
        }

        this.imaginaryCharacter = DEFAULT_IMAGINARY_CHARACTER;
        this.imaginaryFormat = imaginaryFormat;
        this.realFormat = realFormat;
    }

    
    public ComplexFormat(String imaginaryCharacter)
        throws NullArgumentException, NoDataException {
        this(imaginaryCharacter, CompositeFormat.getDefaultNumberFormat());
    }

    
    public ComplexFormat(String imaginaryCharacter, NumberFormat format)
        throws NullArgumentException, NoDataException {
        this(imaginaryCharacter, format, format);
    }

    
    public ComplexFormat(String imaginaryCharacter,
                         NumberFormat realFormat,
                         NumberFormat imaginaryFormat)
        throws NullArgumentException, NoDataException {
        if (imaginaryCharacter == null) {
            throw new NullArgumentException();
        }
        if (imaginaryCharacter.length() == 0) {
            throw new NoDataException();
        }
        if (imaginaryFormat == null) {
            throw new NullArgumentException(LocalizedFormats.IMAGINARY_FORMAT);
        }
        if (realFormat == null) {
            throw new NullArgumentException(LocalizedFormats.REAL_FORMAT);
        }

        this.imaginaryCharacter = imaginaryCharacter;
        this.imaginaryFormat = imaginaryFormat;
        this.realFormat = realFormat;
    }

    
    public static Locale[] getAvailableLocales() {
        return NumberFormat.getAvailableLocales();
    }

    
    public String format(Complex c) {
        return format(c, new StringBuffer(), new FieldPosition(0)).toString();
    }

    
    public String format(Double c) {
        return format(new Complex(c, 0), new StringBuffer(), new FieldPosition(0)).toString();
    }

    
    public StringBuffer format(Complex complex, StringBuffer toAppendTo,
                               FieldPosition pos) {
        pos.setBeginIndex(0);
        pos.setEndIndex(0);

        // format real
        double re = complex.getReal();
        CompositeFormat.formatDouble(re, getRealFormat(), toAppendTo, pos);

        // format sign and imaginary
        double im = complex.getImaginary();
        StringBuffer imAppendTo;
        if (im < 0.0) {
            toAppendTo.append(" - ");
            imAppendTo = formatImaginary(-im, new StringBuffer(), pos);
            toAppendTo.append(imAppendTo);
            toAppendTo.append(getImaginaryCharacter());
        } else if (im > 0.0 || Double.isNaN(im)) {
            toAppendTo.append(" + ");
            imAppendTo = formatImaginary(im, new StringBuffer(), pos);
            toAppendTo.append(imAppendTo);
            toAppendTo.append(getImaginaryCharacter());
        }

        return toAppendTo;
    }

    
    private StringBuffer formatImaginary(double absIm,
                                         StringBuffer toAppendTo,
                                         FieldPosition pos) {
        pos.setBeginIndex(0);
        pos.setEndIndex(0);

        CompositeFormat.formatDouble(absIm, getImaginaryFormat(), toAppendTo, pos);
        if (toAppendTo.toString().equals("1")) {
            // Remove the character "1" if it is the only one.
            toAppendTo.setLength(0);
        }

        return toAppendTo;
    }

    
    public StringBuffer format(Object obj, StringBuffer toAppendTo,
                               FieldPosition pos)
        throws MathIllegalArgumentException {

        StringBuffer ret = null;

        if (obj instanceof Complex) {
            ret = format( (Complex)obj, toAppendTo, pos);
        } else if (obj instanceof Number) {
            ret = format(new Complex(((Number)obj).doubleValue(), 0.0),
                         toAppendTo, pos);
        } else {
            throw new MathIllegalArgumentException(LocalizedFormats.CANNOT_FORMAT_INSTANCE_AS_COMPLEX,
                                                   obj.getClass().getName());
        }

        return ret;
    }

    
    public String getImaginaryCharacter() {
        return imaginaryCharacter;
    }

    
    public NumberFormat getImaginaryFormat() {
        return imaginaryFormat;
    }

    
    public static ComplexFormat getInstance() {
        return getInstance(Locale.getDefault());
    }

    
    public static ComplexFormat getInstance(Locale locale) {
        NumberFormat f = CompositeFormat.getDefaultNumberFormat(locale);
        return new ComplexFormat(f);
    }

    
    public static ComplexFormat getInstance(String imaginaryCharacter, Locale locale)
        throws NullArgumentException, NoDataException {
        NumberFormat f = CompositeFormat.getDefaultNumberFormat(locale);
        return new ComplexFormat(imaginaryCharacter, f);
    }

    
    public NumberFormat getRealFormat() {
        return realFormat;
    }

    
    public Complex parse(String source) throws MathParseException {
        ParsePosition parsePosition = new ParsePosition(0);
        Complex result = parse(source, parsePosition);
        if (parsePosition.getIndex() == 0) {
            throw new MathParseException(source,
                                         parsePosition.getErrorIndex(),
                                         Complex.class);
        }
        return result;
    }

    
    public Complex parse(String source, ParsePosition pos) {
        int initialIndex = pos.getIndex();

        // parse whitespace
        CompositeFormat.parseAndIgnoreWhitespace(source, pos);

        // parse real
        Number re = CompositeFormat.parseNumber(source, getRealFormat(), pos);
        if (re == null) {
            // invalid real number
            // set index back to initial, error index should already be set
            pos.setIndex(initialIndex);
            return null;
        }

        // parse sign
        int startIndex = pos.getIndex();
        char c = CompositeFormat.parseNextCharacter(source, pos);
        int sign = 0;
        switch (c) {
        case 0 :
            // no sign
            // return real only complex number
            return new Complex(re.doubleValue(), 0.0);
        case '-' :
            sign = -1;
            break;
        case '+' :
            sign = 1;
            break;
        default :
            // invalid sign
            // set index back to initial, error index should be the last
            // character examined.
            pos.setIndex(initialIndex);
            pos.setErrorIndex(startIndex);
            return null;
        }

        // parse whitespace
        CompositeFormat.parseAndIgnoreWhitespace(source, pos);

        // parse imaginary
        Number im = CompositeFormat.parseNumber(source, getRealFormat(), pos);
        if (im == null) {
            // invalid imaginary number
            // set index back to initial, error index should already be set
            pos.setIndex(initialIndex);
            return null;
        }

        // parse imaginary character
        if (!CompositeFormat.parseFixedstring(source, getImaginaryCharacter(), pos)) {
            return null;
        }

        return new Complex(re.doubleValue(), im.doubleValue() * sign);

    }
}
