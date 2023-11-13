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
package org.apache.lucene.util.hnsw.math.stat.inference;

import org.apache.lucene.util.hnsw.math.distribution.BinomialDistribution;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class BinomialTest {

    
    public boolean binomialTest(int numberOfTrials, int numberOfSuccesses, double probability,
                                AlternativeHypothesis alternativeHypothesis, double alpha) {
        double pValue = binomialTest(numberOfTrials, numberOfSuccesses, probability, alternativeHypothesis);
        return pValue < alpha;
    }

    
    public double binomialTest(int numberOfTrials, int numberOfSuccesses, double probability,
                               AlternativeHypothesis alternativeHypothesis) {
        if (numberOfTrials < 0) {
            throw new NotPositiveException(numberOfTrials);
        }
        if (numberOfSuccesses < 0) {
            throw new NotPositiveException(numberOfSuccesses);
        }
        if (probability < 0 || probability > 1) {
            throw new OutOfRangeException(probability, 0, 1);
        }
        if (numberOfTrials < numberOfSuccesses) {
            throw new MathIllegalArgumentException(
                LocalizedFormats.BINOMIAL_INVALID_PARAMETERS_ORDER,
                numberOfTrials, numberOfSuccesses);
        }
        if (alternativeHypothesis == null) {
            throw new NullArgumentException();
        }

        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final BinomialDistribution distribution = new BinomialDistribution(null, numberOfTrials, probability);
        switch (alternativeHypothesis) {
        case GREATER_THAN:
            return 1 - distribution.cumulativeProbability(numberOfSuccesses - 1);
        case LESS_THAN:
            return distribution.cumulativeProbability(numberOfSuccesses);
        case TWO_SIDED:
            int criticalValueLow = 0;
            int criticalValueHigh = numberOfTrials;
            double pTotal = 0;

            while (true) {
                double pLow = distribution.probability(criticalValueLow);
                double pHigh = distribution.probability(criticalValueHigh);

                if (pLow == pHigh) {
                    pTotal += 2 * pLow;
                    criticalValueLow++;
                    criticalValueHigh--;
                } else if (pLow < pHigh) {
                    pTotal += pLow;
                    criticalValueLow++;
                } else {
                    pTotal += pHigh;
                    criticalValueHigh--;
                }

                if (criticalValueLow > numberOfSuccesses || criticalValueHigh < numberOfSuccesses) {
                    break;
                }
            }
            return pTotal;
        default:
            throw new MathInternalError(LocalizedFormats. OUT_OF_RANGE_SIMPLE, alternativeHypothesis,
                      AlternativeHypothesis.TWO_SIDED, AlternativeHypothesis.LESS_THAN);
        }
    }
}
