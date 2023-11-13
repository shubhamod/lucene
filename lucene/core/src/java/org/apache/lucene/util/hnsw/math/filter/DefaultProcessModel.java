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
package org.apache.lucene.util.hnsw.math.filter;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NoDataException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;


public class DefaultProcessModel implements ProcessModel {
    
    private RealMatrix stateTransitionMatrix;

    
    private RealMatrix controlMatrix;

    
    private RealMatrix processNoiseCovMatrix;

    
    private RealVector initialStateEstimateVector;

    
    private RealMatrix initialErrorCovMatrix;

    
    public DefaultProcessModel(final double[][] stateTransition,
                               final double[][] control,
                               final double[][] processNoise,
                               final double[] initialStateEstimate,
                               final double[][] initialErrorCovariance)
            throws NullArgumentException, NoDataException, DimensionMismatchException {

        this(new Array2DRowRealMatrix(stateTransition),
                new Array2DRowRealMatrix(control),
                new Array2DRowRealMatrix(processNoise),
                new ArrayRealVector(initialStateEstimate),
                new Array2DRowRealMatrix(initialErrorCovariance));
    }

    
    public DefaultProcessModel(final double[][] stateTransition,
                               final double[][] control,
                               final double[][] processNoise)
            throws NullArgumentException, NoDataException, DimensionMismatchException {

        this(new Array2DRowRealMatrix(stateTransition),
                new Array2DRowRealMatrix(control),
                new Array2DRowRealMatrix(processNoise), null, null);
    }

    
    public DefaultProcessModel(final RealMatrix stateTransition,
                               final RealMatrix control,
                               final RealMatrix processNoise,
                               final RealVector initialStateEstimate,
                               final RealMatrix initialErrorCovariance) {
        this.stateTransitionMatrix = stateTransition;
        this.controlMatrix = control;
        this.processNoiseCovMatrix = processNoise;
        this.initialStateEstimateVector = initialStateEstimate;
        this.initialErrorCovMatrix = initialErrorCovariance;
    }

    
    public RealMatrix getStateTransitionMatrix() {
        return stateTransitionMatrix;
    }

    
    public RealMatrix getControlMatrix() {
        return controlMatrix;
    }

    
    public RealMatrix getProcessNoise() {
        return processNoiseCovMatrix;
    }

    
    public RealVector getInitialStateEstimate() {
        return initialStateEstimateVector;
    }

    
    public RealMatrix getInitialErrorCovariance() {
        return initialErrorCovMatrix;
    }
}
