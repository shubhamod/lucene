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
package org.apache.lucene.util.hnsw.math.stat.correlation;

import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


class StorelessBivariateCovariance {

    
    private double meanX;

    
    private double meanY;

    
    private double n;

    
    private double covarianceNumerator;

    
    private boolean biasCorrected;

    
    StorelessBivariateCovariance() {
        this(true);
    }

    
    StorelessBivariateCovariance(final boolean biasCorrection) {
        meanX = meanY = 0.0;
        n = 0;
        covarianceNumerator = 0.0;
        biasCorrected = biasCorrection;
    }

    
    public void increment(final double x, final double y) {
        n++;
        final double deltaX = x - meanX;
        final double deltaY = y - meanY;
        meanX += deltaX / n;
        meanY += deltaY / n;
        covarianceNumerator += ((n - 1.0) / n) * deltaX * deltaY;
    }

    
    public void append(StorelessBivariateCovariance cov) {
        double oldN = n;
        n += cov.n;
        final double deltaX = cov.meanX - meanX;
        final double deltaY = cov.meanY - meanY;
        meanX += deltaX * cov.n / n;
        meanY += deltaY * cov.n / n;
        covarianceNumerator += cov.covarianceNumerator + oldN * cov.n / n * deltaX * deltaY;
    }

    
    public double getN() {
        return n;
    }

    
    public double getResult() throws NumberIsTooSmallException {
        if (n < 2) {
            throw new NumberIsTooSmallException(LocalizedFormats.INSUFFICIENT_DIMENSION,
                                                n, 2, true);
        }
        if (biasCorrected) {
            return covarianceNumerator / (n - 1d);
        } else {
            return covarianceNumerator / n;
        }
    }
}

