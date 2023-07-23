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

import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.Locale;


public class CompositeFormat {

    
    private CompositeFormat() {}

    
    public static NumberFormat getDefaultNumberFormat() {
        return getDefaultNumberFormat(Locale.getDefault());
    }

    
    public static NumberFormat getDefaultNumberFormat(final Locale locale) {
        final NumberFormat nf = NumberFormat.getInstance(locale);
        nf.setMaximumFractionDigits(10);
        return nf;
    }

    
    public static void parseAndIgnoreWhitespace(final String source,
                                                final ParsePosition pos) {
        parseNextCharacter(source, pos);
        pos.setIndex(pos.getIndex() - 1);
    }

    
    public static char parseNextCharacter(final String source,
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

    
    private static Number parseNumber(final String source, final double value,
                                      final ParsePosition pos) {
        Number ret = null;

        StringBuilder sb = new StringBuilder();
        sb.append('(');
        sb.append(value);
        sb.append(')');

        final int n = sb.length();
        final int startIndex = pos.getIndex();
        final int endIndex = startIndex + n;
        if (endIndex < source.length() &&
            source.substring(startIndex, endIndex).compareTo(sb.toString()) == 0) {
            ret = Double.valueOf(value);
            pos.setIndex(endIndex);
        }

        return ret;
    }

    
    public static Number parseNumber(final String source, final NumberFormat format,
                                     final ParsePosition pos) {
        final int startIndex = pos.getIndex();
        Number number = format.parse(source, pos);
        final int endIndex = pos.getIndex();

        // check for error parsing number
        if (startIndex == endIndex) {
            // try parsing special numbers
            final double[] special = {
                Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY
            };
            for (int i = 0; i < special.length; ++i) {
                number = parseNumber(source, special[i], pos);
                if (number != null) {
                    break;
                }
            }
        }

        return number;
    }

    
    public static boolean parseFixedstring(final String source,
                                           final String expected,
                                           final ParsePosition pos) {

        final int startIndex = pos.getIndex();
        final int endIndex = startIndex + expected.length();
        if ((startIndex >= source.length()) ||
            (endIndex > source.length()) ||
            (source.substring(startIndex, endIndex).compareTo(expected) != 0)) {
            // set index back to start, error index should be the start index
            pos.setIndex(startIndex);
            pos.setErrorIndex(startIndex);
            return false;
        }

        // the string was here
        pos.setIndex(endIndex);
        return true;
    }

    
    public static StringBuffer formatDouble(final double value, final NumberFormat format,
                                            final StringBuffer toAppendTo,
                                            final FieldPosition pos) {
        if( Double.isNaN(value) || Double.isInfinite(value) ) {
            toAppendTo.append('(');
            toAppendTo.append(value);
            toAppendTo.append(')');
        } else {
            format.format(value, toAppendTo, pos);
        }
        return toAppendTo;
    }
}
