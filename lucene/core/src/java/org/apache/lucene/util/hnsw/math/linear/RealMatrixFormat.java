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


public class RealMatrixFormat {

    
    private static final String DEFAULT_PREFIX = "{";
    
    private static final String DEFAULT_SUFFIX = "}";
    
    private static final String DEFAULT_ROW_PREFIX = "{";
    
    private static final String DEFAULT_ROW_SUFFIX = "}";
    
    private static final String DEFAULT_ROW_SEPARATOR = ",";
    
    private static final String DEFAULT_COLUMN_SEPARATOR = ",";
    
    private final String prefix;
    
    private final String suffix;
    
    private final String rowPrefix;
    
    private final String rowSuffix;
    
    private final String rowSeparator;
    
    private final String columnSeparator;
    
    private final NumberFormat format;

    
    public RealMatrixFormat() {
        this(DEFAULT_PREFIX, DEFAULT_SUFFIX, DEFAULT_ROW_PREFIX, DEFAULT_ROW_SUFFIX,
                DEFAULT_ROW_SEPARATOR, DEFAULT_COLUMN_SEPARATOR, CompositeFormat.getDefaultNumberFormat());
    }

    
    public RealMatrixFormat(final NumberFormat format) {
        this(DEFAULT_PREFIX, DEFAULT_SUFFIX, DEFAULT_ROW_PREFIX, DEFAULT_ROW_SUFFIX,
                DEFAULT_ROW_SEPARATOR, DEFAULT_COLUMN_SEPARATOR, format);
    }

    
    public RealMatrixFormat(final String prefix, final String suffix,
                            final String rowPrefix, final String rowSuffix,
                            final String rowSeparator, final String columnSeparator) {
        this(prefix, suffix, rowPrefix, rowSuffix, rowSeparator, columnSeparator,
                CompositeFormat.getDefaultNumberFormat());
    }

    
    public RealMatrixFormat(final String prefix, final String suffix,
                            final String rowPrefix, final String rowSuffix,
                            final String rowSeparator, final String columnSeparator,
                            final NumberFormat format) {
        this.prefix            = prefix;
        this.suffix            = suffix;
        this.rowPrefix         = rowPrefix;
        this.rowSuffix         = rowSuffix;
        this.rowSeparator      = rowSeparator;
        this.columnSeparator   = columnSeparator;
        this.format            = format;
        // disable grouping to prevent parsing problems
        this.format.setGroupingUsed(false);
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

    
    public String getRowPrefix() {
        return rowPrefix;
    }

    
    public String getRowSuffix() {
        return rowSuffix;
    }

    
    public String getRowSeparator() {
        return rowSeparator;
    }

    
    public String getColumnSeparator() {
        return columnSeparator;
    }

    
    public NumberFormat getFormat() {
        return format;
    }

    
    public static RealMatrixFormat getInstance() {
        return getInstance(Locale.getDefault());
    }

    
    public static RealMatrixFormat getInstance(final Locale locale) {
        return new RealMatrixFormat(CompositeFormat.getDefaultNumberFormat(locale));
    }

    
    public String format(RealMatrix m) {
        return format(m, new StringBuffer(), new FieldPosition(0)).toString();
    }

    
    public StringBuffer format(RealMatrix matrix, StringBuffer toAppendTo,
                               FieldPosition pos) {

        pos.setBeginIndex(0);
        pos.setEndIndex(0);

        // format prefix
        toAppendTo.append(prefix);

        // format rows
        final int rows = matrix.getRowDimension();
        for (int i = 0; i < rows; ++i) {
            toAppendTo.append(rowPrefix);
            for (int j = 0; j < matrix.getColumnDimension(); ++j) {
                if (j > 0) {
                    toAppendTo.append(columnSeparator);
                }
                CompositeFormat.formatDouble(matrix.getEntry(i, j), format, toAppendTo, pos);
            }
            toAppendTo.append(rowSuffix);
            if (i < rows - 1) {
                toAppendTo.append(rowSeparator);
            }
        }

        // format suffix
        toAppendTo.append(suffix);

        return toAppendTo;
    }

    
    public RealMatrix parse(String source) {
        final ParsePosition parsePosition = new ParsePosition(0);
        final RealMatrix result = parse(source, parsePosition);
        if (parsePosition.getIndex() == 0) {
            throw new MathParseException(source,
                                         parsePosition.getErrorIndex(),
                                         Array2DRowRealMatrix.class);
        }
        return result;
    }

    
    public RealMatrix parse(String source, ParsePosition pos) {
        int initialIndex = pos.getIndex();

        final String trimmedPrefix = prefix.trim();
        final String trimmedSuffix = suffix.trim();
        final String trimmedRowPrefix = rowPrefix.trim();
        final String trimmedRowSuffix = rowSuffix.trim();
        final String trimmedColumnSeparator = columnSeparator.trim();
        final String trimmedRowSeparator = rowSeparator.trim();

        // parse prefix
        CompositeFormat.parseAndIgnoreWhitespace(source, pos);
        if (!CompositeFormat.parseFixedstring(source, trimmedPrefix, pos)) {
            return null;
        }

        // parse components
        List<List<Number>> matrix = new ArrayList<List<Number>>();
        List<Number> rowComponents = new ArrayList<Number>();
        for (boolean loop = true; loop;){

            if (!rowComponents.isEmpty()) {
                CompositeFormat.parseAndIgnoreWhitespace(source, pos);
                if (!CompositeFormat.parseFixedstring(source, trimmedColumnSeparator, pos)) {
                    if (trimmedRowSuffix.length() != 0 &&
                        !CompositeFormat.parseFixedstring(source, trimmedRowSuffix, pos)) {
                        return null;
                    } else {
                        CompositeFormat.parseAndIgnoreWhitespace(source, pos);
                        if (CompositeFormat.parseFixedstring(source, trimmedRowSeparator, pos)) {
                            matrix.add(rowComponents);
                            rowComponents = new ArrayList<Number>();
                            continue;
                        } else {
                            loop = false;
                        }
                    }
                }
            } else {
                CompositeFormat.parseAndIgnoreWhitespace(source, pos);
                if (trimmedRowPrefix.length() != 0 &&
                    !CompositeFormat.parseFixedstring(source, trimmedRowPrefix, pos)) {
                    return null;
                }
            }

            if (loop) {
                CompositeFormat.parseAndIgnoreWhitespace(source, pos);
                Number component = CompositeFormat.parseNumber(source, format, pos);
                if (component != null) {
                    rowComponents.add(component);
                } else {
                    if (rowComponents.isEmpty()) {
                        loop = false;
                    } else {
                        // invalid component
                        // set index back to initial, error index should already be set
                        pos.setIndex(initialIndex);
                        return null;
                    }
                }
            }

        }

        if (!rowComponents.isEmpty()) {
            matrix.add(rowComponents);
        }

        // parse suffix
        CompositeFormat.parseAndIgnoreWhitespace(source, pos);
        if (!CompositeFormat.parseFixedstring(source, trimmedSuffix, pos)) {
            return null;
        }

        // do not allow an empty matrix
        if (matrix.isEmpty()) {
            pos.setIndex(initialIndex);
            return null;
        }

        // build vector
        double[][] data = new double[matrix.size()][];
        int row = 0;
        for (List<Number> rowList : matrix) {
            data[row] = new double[rowList.size()];
            for (int i = 0; i < rowList.size(); i++) {
                data[row][i] = rowList.get(i).doubleValue();
            }
            row++;
        }
        return MatrixUtils.createRealMatrix(data);
    }
}
