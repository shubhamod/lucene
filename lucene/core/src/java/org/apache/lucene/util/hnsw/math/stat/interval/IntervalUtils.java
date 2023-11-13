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

import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public final class IntervalUtils {

    
    private static final BinomialConfidenceInterval AGRESTI_COULL = new AgrestiCoullInterval();

    
    private static final BinomialConfidenceInterval CLOPPER_PEARSON = new ClopperPearsonInterval();

    
    private static final BinomialConfidenceInterval NORMAL_APPROXIMATION = new NormalApproximationInterval();

    
    private static final BinomialConfidenceInterval WILSON_SCORE = new WilsonScoreInterval();

    
    private IntervalUtils() {
    }

    
    public static ConfidenceInterval getAgrestiCoullInterval(int numberOfTrials, int numberOfSuccesses,
                                                             double confidenceLevel) {
        return AGRESTI_COULL.createInterval(numberOfTrials, numberOfSuccesses, confidenceLevel);
    }

    
    public static ConfidenceInterval getClopperPearsonInterval(int numberOfTrials, int numberOfSuccesses,
                                                               double confidenceLevel) {
        return CLOPPER_PEARSON.createInterval(numberOfTrials, numberOfSuccesses, confidenceLevel);
    }

    
    public static ConfidenceInterval getNormalApproximationInterval(int numberOfTrials, int numberOfSuccesses,
                                                                    double confidenceLevel) {
        return NORMAL_APPROXIMATION.createInterval(numberOfTrials, numberOfSuccesses, confidenceLevel);
    }

    
    public static ConfidenceInterval getWilsonScoreInterval(int numberOfTrials, int numberOfSuccesses,
                                                            double confidenceLevel) {
        return WILSON_SCORE.createInterval(numberOfTrials, numberOfSuccesses, confidenceLevel);
    }

    
    static void checkParameters(int numberOfTrials, int numberOfSuccesses, double confidenceLevel) {
        if (numberOfTrials <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_TRIALS, numberOfTrials);
        }
        if (numberOfSuccesses < 0) {
            throw new NotPositiveException(LocalizedFormats.NEGATIVE_NUMBER_OF_SUCCESSES, numberOfSuccesses);
        }
        if (numberOfSuccesses > numberOfTrials) {
            throw new NumberIsTooLargeException(LocalizedFormats.NUMBER_OF_SUCCESS_LARGER_THAN_POPULATION_SIZE,
                                                numberOfSuccesses, numberOfTrials, true);
        }
        if (confidenceLevel <= 0 || confidenceLevel >= 1) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_BOUNDS_CONFIDENCE_LEVEL,
                                          confidenceLevel, 0, 1);
        }
    }

}
