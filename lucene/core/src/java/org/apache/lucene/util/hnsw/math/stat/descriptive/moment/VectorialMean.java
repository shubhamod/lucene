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


public class VectorialMean implements Serializable {

    
    private static final long serialVersionUID = 8223009086481006892L;

    
    private final Mean[] means;

    
    public VectorialMean(int dimension) {
        means = new Mean[dimension];
        for (int i = 0; i < dimension; ++i) {
            means[i] = new Mean();
        }
    }

    
    public void increment(double[] v) throws DimensionMismatchException {
        if (v.length != means.length) {
            throw new DimensionMismatchException(v.length, means.length);
        }
        for (int i = 0; i < v.length; ++i) {
            means[i].increment(v[i]);
        }
    }

    
    public double[] getResult() {
        double[] result = new double[means.length];
        for (int i = 0; i < result.length; ++i) {
            result[i] = means[i].getResult();
        }
        return result;
    }

    
    public long getN() {
        return (means.length == 0) ? 0 : means[0].getN();
    }

    
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + Arrays.hashCode(means);
        return result;
    }

    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof VectorialMean)) {
            return false;
        }
        VectorialMean other = (VectorialMean) obj;
        if (!Arrays.equals(means, other.means)) {
            return false;
        }
        return true;
    }

}
