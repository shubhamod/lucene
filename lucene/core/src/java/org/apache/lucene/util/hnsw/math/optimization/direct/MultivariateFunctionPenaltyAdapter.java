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

package org.apache.lucene.util.hnsw.math.optimization.direct;

import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;



@Deprecated
public class MultivariateFunctionPenaltyAdapter implements MultivariateFunction {

    
    private final MultivariateFunction bounded;

    
    private final double[] lower;

    
    private final double[] upper;

    
    private final double offset;

    
    private final double[] scale;

    
    public MultivariateFunctionPenaltyAdapter(final MultivariateFunction bounded,
                                                  final double[] lower, final double[] upper,
                                                  final double offset, final double[] scale) {

        // safety checks
        MathUtils.checkNotNull(lower);
        MathUtils.checkNotNull(upper);
        MathUtils.checkNotNull(scale);
        if (lower.length != upper.length) {
            throw new DimensionMismatchException(lower.length, upper.length);
        }
        if (lower.length != scale.length) {
            throw new DimensionMismatchException(lower.length, scale.length);
        }
        for (int i = 0; i < lower.length; ++i) {
            // note the following test is written in such a way it also fails for NaN
            if (!(upper[i] >= lower[i])) {
                throw new NumberIsTooSmallException(upper[i], lower[i], true);
            }
        }

        this.bounded = bounded;
        this.lower   = lower.clone();
        this.upper   = upper.clone();
        this.offset  = offset;
        this.scale   = scale.clone();

    }

    
    public double value(double[] point) {

        for (int i = 0; i < scale.length; ++i) {
            if ((point[i] < lower[i]) || (point[i] > upper[i])) {
                // bound violation starting at this component
                double sum = 0;
                for (int j = i; j < scale.length; ++j) {
                    final double overshoot;
                    if (point[j] < lower[j]) {
                        overshoot = scale[j] * (lower[j] - point[j]);
                    } else if (point[j] > upper[j]) {
                        overshoot = scale[j] * (point[j] - upper[j]);
                    } else {
                        overshoot = 0;
                    }
                    sum += FastMath.sqrt(overshoot);
                }
                return offset + sum;
            }
        }

        // all boundaries are fulfilled, we are in the expected
        // domain of the underlying function
        return bounded.value(point);

    }

}
