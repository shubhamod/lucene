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
package org.apache.lucene.util.hnsw.math.stat.interval;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class ConfidenceInterval {

    
    private double lowerBound;

    
    private double upperBound;

    
    private double confidenceLevel;

    
    public ConfidenceInterval(double lowerBound, double upperBound, double confidenceLevel) {
        checkParameters(lowerBound, upperBound, confidenceLevel);
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.confidenceLevel = confidenceLevel;
    }

    
    public double getLowerBound() {
        return lowerBound;
    }

    
    public double getUpperBound() {
        return upperBound;
    }

    
    public double getConfidenceLevel() {
        return confidenceLevel;
    }

    
    @Override
    public String toString() {
        return "[" + lowerBound + ";" + upperBound + "] (confidence level:" + confidenceLevel + ")";
    }

    
    private void checkParameters(double lower, double upper, double confidence) {
        if (lower >= upper) {
            throw new MathIllegalArgumentException(LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND, lower, upper);
        }
        if (confidence <= 0 || confidence >= 1) {
            throw new MathIllegalArgumentException(LocalizedFormats.OUT_OF_BOUNDS_CONFIDENCE_LEVEL, confidence, 0, 1);
        }
    }
}
