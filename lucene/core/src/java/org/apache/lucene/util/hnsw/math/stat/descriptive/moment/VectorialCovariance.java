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
package org.apache.lucene.util.hnsw.math.stat.descriptive.moment;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;


public class VectorialCovariance implements Serializable {

    
    private static final long serialVersionUID = 4118372414238930270L;

    
    private final double[] sums;

    
    private final double[] productsSums;

    
    private final boolean isBiasCorrected;

    
    private long n;

    
    public VectorialCovariance(int dimension, boolean isBiasCorrected) {
        sums         = new double[dimension];
        productsSums = new double[dimension * (dimension + 1) / 2];
        n            = 0;
        this.isBiasCorrected = isBiasCorrected;
    }

    
    public void increment(double[] v) throws DimensionMismatchException {
        if (v.length != sums.length) {
            throw new DimensionMismatchException(v.length, sums.length);
        }
        int k = 0;
        for (int i = 0; i < v.length; ++i) {
            sums[i] += v[i];
            for (int j = 0; j <= i; ++j) {
                productsSums[k++] += v[i] * v[j];
            }
        }
        n++;
    }

    
    public RealMatrix getResult() {

        int dimension = sums.length;
        RealMatrix result = MatrixUtils.createRealMatrix(dimension, dimension);

        if (n > 1) {
            double c = 1.0 / (n * (isBiasCorrected ? (n - 1) : n));
            int k = 0;
            for (int i = 0; i < dimension; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double e = c * (n * productsSums[k++] - sums[i] * sums[j]);
                    result.setEntry(i, j, e);
                    result.setEntry(j, i, e);
                }
            }
        }

        return result;

    }

    
    public long getN() {
        return n;
    }

    
    public void clear() {
        n = 0;
        Arrays.fill(sums, 0.0);
        Arrays.fill(productsSums, 0.0);
    }

    
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + (isBiasCorrected ? 1231 : 1237);
        result = prime * result + (int) (n ^ (n >>> 32));
        result = prime * result + Arrays.hashCode(productsSums);
        result = prime * result + Arrays.hashCode(sums);
        return result;
    }

    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof VectorialCovariance)) {
            return false;
        }
        VectorialCovariance other = (VectorialCovariance) obj;
        if (isBiasCorrected != other.isBiasCorrected) {
            return false;
        }
        if (n != other.n) {
            return false;
        }
        if (!Arrays.equals(productsSums, other.productsSums)) {
            return false;
        }
        if (!Arrays.equals(sums, other.sums)) {
            return false;
        }
        return true;
    }

}
