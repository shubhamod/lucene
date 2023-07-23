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

package org.apache.lucene.util.hnsw.math.linear;

import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.apache.lucene.util.hnsw.math.exception.MathParseException;
import org.apache.lucene.util.hnsw.math.util.CompositeFormat;


public class RealVectorFormat {

    
    private static final String DEFAULT_PREFIX = "{";
    
    private static final String DEFAULT_SUFFIX = "}";
    
    private static final String DEFAULT_SEPARATOR = "; ";
    
    private final String prefix;
    
    private final String suffix;
    
    private final String separator;
    
    private final String trimmedPrefix;
    
    private final String trimmedSuffix;
    
    private final String trimmedSeparator;
    
    private final NumberFormat format;

    
    public RealVectorFormat() {
        this(DEFAULT_PREFIX, DEFAULT_SUFFIX, DEFAULT_SEPARATOR,
             CompositeFormat.getDefaultNumberFormat());
    }

    
    public RealVectorFormat(final NumberFormat format) {
        this(DEFAULT_PREFIX, DEFAULT_SUFFIX, DEFAULT_SEPARATOR, format);
    }

    
    public RealVectorFormat(final String prefix, final String suffix,
                            final String separator) {
        this(prefix, suffix, separator,
             CompositeFormat.getDefaultNumberFormat());
    }

    
    public RealVectorFormat(final String prefix, final String suffix,
                            final String separator, final NumberFormat format) {
        this.prefix      = prefix;
        this.suffix      = suffix;
        this.separator   = separator;
        trimmedPrefix    = prefix.trim();
        trimmedSuffix    = suffix.trim();
        trimmedSeparator = separator.trim();
        this.format      = format;
    }

    
    public static Locale[] getAvailableLocales() {
        return NumberFormat.getAvailableLocales();
    }

    
    public String getPrefix() {
        return prefix;
    }

    
    public String getSuffix() {
        return suffix;
    }

    
    public String getSeparator() {
        return separator;
    }

    
    public NumberFormat getFormat() {
        return format;
    }

    
    public static RealVectorFormat getInstance() {
        return getInstance(Locale.getDefault());
    }

    
    public static RealVectorFormat getInstance(final Locale locale) {
        return new RealVectorFormat(CompositeFormat.getDefaultNumberFormat(locale));
    }

    
    public String format(RealVector v) {
        return format(v, new StringBuffer(), new FieldPosition(0)).toString();
    }

    
    public StringBuffer format(RealVector vector, StringBuffer toAppendTo,
                               FieldPosition pos) {

        pos.setBeginIndex(0);
        pos.setEndIndex(0);

        // format prefix
        toAppendTo.append(prefix);

        // format components
        for (int i = 0; i < vector.getDimension(); ++i) {
            if (i > 0) {
                toAppendTo.append(separator);
            }
            CompositeFormat.formatDouble(vector.getEntry(i), format, toAppendTo, pos);
        }

        // format suffix
        toAppendTo.append(suffix);

        return toAppendTo;
    }

    
    public ArrayRealVector parse(String source) {
        final ParsePosition parsePosition = new ParsePosition(0);
        final ArrayRealVector result = parse(source, parsePosition);
        if (parsePosition.getIndex() == 0) {
            throw new MathParseException(source,
                                         parsePosition.getErrorIndex(),
                                         ArrayRealVector.class);
        }
        return result;
    }

    
    public ArrayRealVector parse(String source, ParsePosition pos) {
        int initialIndex = pos.getIndex();

        // parse prefix
        CompositeFormat.parseAndIgnoreWhitespace(source, pos);
        if (!CompositeFormat.parseFixedstring(source, trimmedPrefix, pos)) {
            return null;
        }

        // parse components
        List<Number> components = new ArrayList<Number>();
        for (boolean loop = true; loop;){

            if (!components.isEmpty()) {
                CompositeFormat.parseAndIgnoreWhitespace(source, pos);
                if (!CompositeFormat.parseFixedstring(source, trimmedSeparator, pos)) {
                    loop = false;
                }
            }

            if (loop) {
                CompositeFormat.parseAndIgnoreWhitespace(source, pos);
                Number component = CompositeFormat.parseNumber(source, format, pos);
                if (component != null) {
                    components.add(component);
                } else {
                    // invalid component
                    // set index back to initial, error index should already be set
                    pos.setIndex(initialIndex);
                    return null;
                }
            }

        }

        // parse suffix
        CompositeFormat.parseAndIgnoreWhitespace(source, pos);
        if (!CompositeFormat.parseFixedstring(source, trimmedSuffix, pos)) {
            return null;
        }

        // build vector
        double[] data = new double[components.size()];
        for (int i = 0; i < data.length; ++i) {
            data[i] = components.get(i).doubleValue();
        }
        return new ArrayRealVector(data, false);
    }
}
