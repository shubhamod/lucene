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

package org.apache.lucene.util.hnsw.math.analysis.function;

import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class StepFunction implements UnivariateFunction {
    
    private final double[] abscissa;
    
    private final double[] ordinate;

    
    public StepFunction(double[] x,
                        double[] y)
        throws NullArgumentException, NoDataException,
               DimensionMismatchException, NonMonotonicSequenceException {
        if (x == null ||
            y == null) {
            throw new NullArgumentException();
        }
        if (x.length == 0 ||
            y.length == 0) {
            throw new NoDataException();
        }
        if (y.length != x.length) {
            throw new DimensionMismatchException(y.length, x.length);
        }
        MathArrays.checkOrder(x);

        abscissa = MathArrays.copyOf(x);
        ordinate = MathArrays.copyOf(y);
    }

    
    public double value(double x) {
        int index = Arrays.binarySearch(abscissa, x);
        double fx = 0;

        if (index < -1) {
            // "x" is between "abscissa[-index-2]" and "abscissa[-index-1]".
            fx = ordinate[-index-2];
        } else if (index >= 0) {
            // "x" is exactly "abscissa[index]".
            fx = ordinate[index];
        } else {
            // Otherwise, "x" is smaller than the first value in "abscissa"
            // (hence the returned value should be "ordinate[0]").
            fx = ordinate[0];
        }

        return fx;
    }
}
