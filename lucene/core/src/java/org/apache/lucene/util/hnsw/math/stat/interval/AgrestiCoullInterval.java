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

import org.apache.lucene.util.hnsw.math.distribution.NormalDistribution;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class AgrestiCoullInterval implements BinomialConfidenceInterval {

    
    public ConfidenceInterval createInterval(int numberOfTrials, int numberOfSuccesses, double confidenceLevel) {
        IntervalUtils.checkParameters(numberOfTrials, numberOfSuccesses, confidenceLevel);
        final double alpha = (1.0 - confidenceLevel) / 2;
        final NormalDistribution normalDistribution = new NormalDistribution();
        final double z = normalDistribution.inverseCumulativeProbability(1 - alpha);
        final double zSquared = FastMath.pow(z, 2);
        final double modifiedNumberOfTrials = numberOfTrials + zSquared;
        final double modifiedSuccessesRatio = (1.0 / modifiedNumberOfTrials) * (numberOfSuccesses + 0.5 * zSquared);
        final double difference = z *
                                  FastMath.sqrt(1.0 / modifiedNumberOfTrials * modifiedSuccessesRatio *
                                                (1 - modifiedSuccessesRatio));
        return new ConfidenceInterval(modifiedSuccessesRatio - difference, modifiedSuccessesRatio + difference,
                                      confidenceLevel);
    }

}
