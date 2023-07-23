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
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.function.Logit;
import org.apache.lucene.util.hnsw.math.analysis.function.Sigmoid;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;



@Deprecated
public class MultivariateFunctionMappingAdapter implements MultivariateFunction {

    
    private final MultivariateFunction bounded;

    
    private final Mapper[] mappers;

    
    public MultivariateFunctionMappingAdapter(final MultivariateFunction bounded,
                                                  final double[] lower, final double[] upper) {

        // safety checks
        MathUtils.checkNotNull(lower);
        MathUtils.checkNotNull(upper);
        if (lower.length != upper.length) {
            throw new DimensionMismatchException(lower.length, upper.length);
        }
        for (int i = 0; i < lower.length; ++i) {
            // note the following test is written in such a way it also fails for NaN
            if (!(upper[i] >= lower[i])) {
                throw new NumberIsTooSmallException(upper[i], lower[i], true);
            }
        }

        this.bounded = bounded;
        this.mappers = new Mapper[lower.length];
        for (int i = 0; i < mappers.length; ++i) {
            if (Double.isInfinite(lower[i])) {
                if (Double.isInfinite(upper[i])) {
                    // element is unbounded, no transformation is needed
                    mappers[i] = new NoBoundsMapper();
                } else {
                    // element is simple-bounded on the upper side
                    mappers[i] = new UpperBoundMapper(upper[i]);
                }
            } else {
                if (Double.isInfinite(upper[i])) {
                    // element is simple-bounded on the lower side
                    mappers[i] = new LowerBoundMapper(lower[i]);
                } else {
                    // element is double-bounded
                    mappers[i] = new LowerUpperBoundMapper(lower[i], upper[i]);
                }
            }
        }

    }

    
    public double[] unboundedToBounded(double[] point) {

        // map unbounded input point to bounded point
        final double[] mapped = new double[mappers.length];
        for (int i = 0; i < mappers.length; ++i) {
            mapped[i] = mappers[i].unboundedToBounded(point[i]);
        }

        return mapped;

    }

    
    public double[] boundedToUnbounded(double[] point) {

        // map bounded input point to unbounded point
        final double[] mapped = new double[mappers.length];
        for (int i = 0; i < mappers.length; ++i) {
            mapped[i] = mappers[i].boundedToUnbounded(point[i]);
        }

        return mapped;

    }

    
    public double value(double[] point) {
        return bounded.value(unboundedToBounded(point));
    }

    
    private interface Mapper {

        
        double unboundedToBounded(double y);

        
        double boundedToUnbounded(double x);

    }

    
    private static class NoBoundsMapper implements Mapper {

        
        NoBoundsMapper() {
        }

        
        public double unboundedToBounded(final double y) {
            return y;
        }

        
        public double boundedToUnbounded(final double x) {
            return x;
        }

    }

    
    private static class LowerBoundMapper implements Mapper {

        
        private final double lower;

        
        LowerBoundMapper(final double lower) {
            this.lower = lower;
        }

        
        public double unboundedToBounded(final double y) {
            return lower + FastMath.exp(y);
        }

        
        public double boundedToUnbounded(final double x) {
            return FastMath.log(x - lower);
        }

    }

    
    private static class UpperBoundMapper implements Mapper {

        
        private final double upper;

        
        UpperBoundMapper(final double upper) {
            this.upper = upper;
        }

        
        public double unboundedToBounded(final double y) {
            return upper - FastMath.exp(-y);
        }

        
        public double boundedToUnbounded(final double x) {
            return -FastMath.log(upper - x);
        }

    }

    
    private static class LowerUpperBoundMapper implements Mapper {

        
        private final UnivariateFunction boundingFunction;

        
        private final UnivariateFunction unboundingFunction;

        
        LowerUpperBoundMapper(final double lower, final double upper) {
            boundingFunction   = new Sigmoid(lower, upper);
            unboundingFunction = new Logit(lower, upper);
        }

        
        public double unboundedToBounded(final double y) {
            return boundingFunction.value(y);
        }

        
        public double boundedToUnbounded(final double x) {
            return unboundingFunction.value(x);
        }

    }

}
