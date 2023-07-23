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
package org.apache.lucene.util.hnsw.math.complex;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class RootsOfUnity implements Serializable {

    
    private static final long serialVersionUID = 20120201L;

    
    private int omegaCount;

    
    private double[] omegaReal;

    
    private double[] omegaImaginaryCounterClockwise;

    
    private double[] omegaImaginaryClockwise;

    
    private boolean isCounterClockWise;

    
    public RootsOfUnity() {

        omegaCount = 0;
        omegaReal = null;
        omegaImaginaryCounterClockwise = null;
        omegaImaginaryClockwise = null;
        isCounterClockWise = true;
    }

    
    public synchronized boolean isCounterClockWise()
            throws MathIllegalStateException {

        if (omegaCount == 0) {
            throw new MathIllegalStateException(
                    LocalizedFormats.ROOTS_OF_UNITY_NOT_COMPUTED_YET);
        }
        return isCounterClockWise;
    }

    
    public synchronized void computeRoots(int n) throws ZeroException {

        if (n == 0) {
            throw new ZeroException(
                    LocalizedFormats.CANNOT_COMPUTE_0TH_ROOT_OF_UNITY);
        }

        isCounterClockWise = n > 0;

        // avoid repetitive calculations
        final int absN = FastMath.abs(n);

        if (absN == omegaCount) {
            return;
        }

        // calculate everything from scratch
        final double t = 2.0 * FastMath.PI / absN;
        final double cosT = FastMath.cos(t);
        final double sinT = FastMath.sin(t);
        omegaReal = new double[absN];
        omegaImaginaryCounterClockwise = new double[absN];
        omegaImaginaryClockwise = new double[absN];
        omegaReal[0] = 1.0;
        omegaImaginaryCounterClockwise[0] = 0.0;
        omegaImaginaryClockwise[0] = 0.0;
        for (int i = 1; i < absN; i++) {
            omegaReal[i] = omegaReal[i - 1] * cosT -
                    omegaImaginaryCounterClockwise[i - 1] * sinT;
            omegaImaginaryCounterClockwise[i] = omegaReal[i - 1] * sinT +
                    omegaImaginaryCounterClockwise[i - 1] * cosT;
            omegaImaginaryClockwise[i] = -omegaImaginaryCounterClockwise[i];
        }
        omegaCount = absN;
    }

    
    public synchronized double getReal(int k)
            throws MathIllegalStateException, MathIllegalArgumentException {

        if (omegaCount == 0) {
            throw new MathIllegalStateException(
                    LocalizedFormats.ROOTS_OF_UNITY_NOT_COMPUTED_YET);
        }
        if ((k < 0) || (k >= omegaCount)) {
            throw new OutOfRangeException(
                    LocalizedFormats.OUT_OF_RANGE_ROOT_OF_UNITY_INDEX,
                    Integer.valueOf(k),
                    Integer.valueOf(0),
                    Integer.valueOf(omegaCount - 1));
        }

        return omegaReal[k];
    }

    
    public synchronized double getImaginary(int k)
            throws MathIllegalStateException, OutOfRangeException {

        if (omegaCount == 0) {
            throw new MathIllegalStateException(
                    LocalizedFormats.ROOTS_OF_UNITY_NOT_COMPUTED_YET);
        }
        if ((k < 0) || (k >= omegaCount)) {
            throw new OutOfRangeException(
                    LocalizedFormats.OUT_OF_RANGE_ROOT_OF_UNITY_INDEX,
                    Integer.valueOf(k),
                    Integer.valueOf(0),
                    Integer.valueOf(omegaCount - 1));
        }

        return isCounterClockWise ? omegaImaginaryCounterClockwise[k] :
            omegaImaginaryClockwise[k];
    }

    
    public synchronized int getNumberOfRoots() {
        return omegaCount;
    }
}
