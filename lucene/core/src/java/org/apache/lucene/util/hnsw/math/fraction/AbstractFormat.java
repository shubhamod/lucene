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
import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.Locale;

import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public abstract class AbstractFormat extends NumberFormat implements Serializable {

    
    private static final long serialVersionUID = -6981118387974191891L;

    
    private NumberFormat denominatorFormat;

    
    private NumberFormat numeratorFormat;

    
    protected AbstractFormat() {
        this(getDefaultNumberFormat());
    }

    
    protected AbstractFormat(final NumberFormat format) {
        this(format, (NumberFormat) format.clone());
    }

    
    protected AbstractFormat(final NumberFormat numeratorFormat,
                             final NumberFormat denominatorFormat) {
        this.numeratorFormat   = numeratorFormat;
        this.denominatorFormat = denominatorFormat;
    }

    
    protected static NumberFormat getDefaultNumberFormat() {
        return getDefaultNumberFormat(Locale.getDefault());
    }

    
    protected static NumberFormat getDefaultNumberFormat(final Locale locale) {
        final NumberFormat nf = NumberFormat.getNumberInstance(locale);
        nf.setMaximumFractionDigits(0);
        nf.setParseIntegerOnly(true);
        return nf;
    }

    
    public NumberFormat getDenominatorFormat() {
        return denominatorFormat;
    }

    
    public NumberFormat getNumeratorFormat() {
        return numeratorFormat;
    }

    
    public void setDenominatorFormat(final NumberFormat format) {
        if (format == null) {
            throw new NullArgumentException(LocalizedFormats.DENOMINATOR_FORMAT);
        }
        this.denominatorFormat = format;
    }

    
    public void setNumeratorFormat(final NumberFormat format) {
        if (format == null) {
            throw new NullArgumentException(LocalizedFormats.NUMERATOR_FORMAT);
        }
        this.numeratorFormat = format;
    }

    
    protected static void parseAndIgnoreWhitespace(final String source,
                                                   final ParsePosition pos) {
        parseNextCharacter(source, pos);
        pos.setIndex(pos.getIndex() - 1);
    }

    
    protected static char parseNextCharacter(final String source,
                                             final ParsePosition pos) {
         int index = pos.getIndex();
         final int n = source.length();
         char ret = 0;

         if (index < n) {
             char c;
             do {
                 c = source.charAt(index++);
             } while (Character.isWhitespace(c) && index < n);
             pos.setIndex(index);

             if (index < n) {
                 ret = c;
             }
         }

         return ret;
    }

    
    @Override
    public StringBuffer format(final double value,
                               final StringBuffer buffer, final FieldPosition position) {
        return format(Double.valueOf(value), buffer, position);
    }


    
    @Override
    public StringBuffer format(final long value,
                               final StringBuffer buffer, final FieldPosition position) {
        return format(Long.valueOf(value), buffer, position);
    }

}
