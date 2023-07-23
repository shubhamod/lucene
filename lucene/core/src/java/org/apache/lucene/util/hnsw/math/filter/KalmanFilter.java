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
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.CholeskyDecomposition;
import org.apache.lucene.util.hnsw.math.linear.MatrixDimensionMismatchException;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.NonSquareMatrixException;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.linear.SingularMatrixException;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class KalmanFilter {
    
    private final ProcessModel processModel;
    
    private final MeasurementModel measurementModel;
    
    private RealMatrix transitionMatrix;
    
    private RealMatrix transitionMatrixT;
    
    private RealMatrix controlMatrix;
    
    private RealMatrix measurementMatrix;
    
    private RealMatrix measurementMatrixT;
    
    private RealVector stateEstimation;
    
    private RealMatrix errorCovariance;

    
    public KalmanFilter(final ProcessModel process, final MeasurementModel measurement)
            throws NullArgumentException, NonSquareMatrixException, DimensionMismatchException,
                   MatrixDimensionMismatchException {

        MathUtils.checkNotNull(process);
        MathUtils.checkNotNull(measurement);

        this.processModel = process;
        this.measurementModel = measurement;

        transitionMatrix = processModel.getStateTransitionMatrix();
        MathUtils.checkNotNull(transitionMatrix);
        transitionMatrixT = transitionMatrix.transpose();

        // create an empty matrix if no control matrix was given
        if (processModel.getControlMatrix() == null) {
            controlMatrix = new Array2DRowRealMatrix();
        } else {
            controlMatrix = processModel.getControlMatrix();
        }

        measurementMatrix = measurementModel.getMeasurementMatrix();
        MathUtils.checkNotNull(measurementMatrix);
        measurementMatrixT = measurementMatrix.transpose();

        // check that the process and measurement noise matrices are not null
        // they will be directly accessed from the model as they may change
        // over time
        RealMatrix processNoise = processModel.getProcessNoise();
        MathUtils.checkNotNull(processNoise);
        RealMatrix measNoise = measurementModel.getMeasurementNoise();
        MathUtils.checkNotNull(measNoise);

        // set the initial state estimate to a zero vector if it is not
        // available from the process model
        if (processModel.getInitialStateEstimate() == null) {
            stateEstimation = new ArrayRealVector(transitionMatrix.getColumnDimension());
        } else {
            stateEstimation = processModel.getInitialStateEstimate();
        }

        if (transitionMatrix.getColumnDimension() != stateEstimation.getDimension()) {
            throw new DimensionMismatchException(transitionMatrix.getColumnDimension(),
                                                 stateEstimation.getDimension());
        }

        // initialize the error covariance to the process noise if it is not
        // available from the process model
        if (processModel.getInitialErrorCovariance() == null) {
            errorCovariance = processNoise.copy();
        } else {
            errorCovariance = processModel.getInitialErrorCovariance();
        }

        // sanity checks, the control matrix B may be null

        // A must be a square matrix
        if (!transitionMatrix.isSquare()) {
            throw new NonSquareMatrixException(
                    transitionMatrix.getRowDimension(),
                    transitionMatrix.getColumnDimension());
        }

        // row dimension of B must be equal to A
        // if no control matrix is available, the row and column dimension will be 0
        if (controlMatrix != null &&
            controlMatrix.getRowDimension() > 0 &&
            controlMatrix.getColumnDimension() > 0 &&
            controlMatrix.getRowDimension() != transitionMatrix.getRowDimension()) {
            throw new MatrixDimensionMismatchException(controlMatrix.getRowDimension(),
                                                       controlMatrix.getColumnDimension(),
                                                       transitionMatrix.getRowDimension(),
                                                       controlMatrix.getColumnDimension());
        }

        // Q must be equal to A
        MatrixUtils.checkAdditionCompatible(transitionMatrix, processNoise);

        // column dimension of H must be equal to row dimension of A
        if (measurementMatrix.getColumnDimension() != transitionMatrix.getRowDimension()) {
            throw new MatrixDimensionMismatchException(measurementMatrix.getRowDimension(),
                                                       measurementMatrix.getColumnDimension(),
                                                       measurementMatrix.getRowDimension(),
                                                       transitionMatrix.getRowDimension());
        }

        // row dimension of R must be equal to row dimension of H
        if (measNoise.getRowDimension() != measurementMatrix.getRowDimension()) {
            throw new MatrixDimensionMismatchException(measNoise.getRowDimension(),
                                                       measNoise.getColumnDimension(),
                                                       measurementMatrix.getRowDimension(),
                                                       measNoise.getColumnDimension());
        }
    }

    
    public int getStateDimension() {
        return stateEstimation.getDimension();
    }

    
    public int getMeasurementDimension() {
        return measurementMatrix.getRowDimension();
    }

    
    public double[] getStateEstimation() {
        return stateEstimation.toArray();
    }

    
    public RealVector getStateEstimationVector() {
        return stateEstimation.copy();
    }

    
    public double[][] getErrorCovariance() {
        return errorCovariance.getData();
    }

    
    public RealMatrix getErrorCovarianceMatrix() {
        return errorCovariance.copy();
    }

    
    public void predict() {
        predict((RealVector) null);
    }

    
    public void predict(final double[] u) throws DimensionMismatchException {
        predict(new ArrayRealVector(u, false));
    }

    
    public void predict(final RealVector u) throws DimensionMismatchException {
        // sanity checks
        if (u != null &&
            u.getDimension() != controlMatrix.getColumnDimension()) {
            throw new DimensionMismatchException(u.getDimension(),
                                                 controlMatrix.getColumnDimension());
        }

        // project the state estimation ahead (a priori state)
        // xHat(k)- = A * xHat(k-1) + B * u(k-1)
        stateEstimation = transitionMatrix.operate(stateEstimation);

        // add control input if it is available
        if (u != null) {
            stateEstimation = stateEstimation.add(controlMatrix.operate(u));
        }

        // project the error covariance ahead
        // P(k)- = A * P(k-1) * A' + Q
        errorCovariance = transitionMatrix.multiply(errorCovariance)
                .multiply(transitionMatrixT)
                .add(processModel.getProcessNoise());
    }

    
    public void correct(final double[] z)
            throws NullArgumentException, DimensionMismatchException, SingularMatrixException {
        correct(new ArrayRealVector(z, false));
    }

    
    public void correct(final RealVector z)
            throws NullArgumentException, DimensionMismatchException, SingularMatrixException {

        // sanity checks
        MathUtils.checkNotNull(z);
        if (z.getDimension() != measurementMatrix.getRowDimension()) {
            throw new DimensionMismatchException(z.getDimension(),
                                                 measurementMatrix.getRowDimension());
        }

        // S = H * P(k) * H' + R
        RealMatrix s = measurementMatrix.multiply(errorCovariance)
            .multiply(measurementMatrixT)
            .add(measurementModel.getMeasurementNoise());

        // Inn = z(k) - H * xHat(k)-
        RealVector innovation = z.subtract(measurementMatrix.operate(stateEstimation));

        // calculate gain matrix
        // K(k) = P(k)- * H' * (H * P(k)- * H' + R)^-1
        // K(k) = P(k)- * H' * S^-1

        // instead of calculating the inverse of S we can rearrange the formula,
        // and then solve the linear equation A x X = B with A = S', X = K' and B = (H * P)'

        // K(k) * S = P(k)- * H'
        // S' * K(k)' = H * P(k)-'
        RealMatrix kalmanGain = new CholeskyDecomposition(s).getSolver()
                .solve(measurementMatrix.multiply(errorCovariance.transpose()))
                .transpose();

        // update estimate with measurement z(k)
        // xHat(k) = xHat(k)- + K * Inn
        stateEstimation = stateEstimation.add(kalmanGain.operate(innovation));

        // update covariance of prediction error
        // P(k) = (I - K * H) * P(k)-
        RealMatrix identity = MatrixUtils.createRealIdentityMatrix(kalmanGain.getRowDimension());
        errorCovariance = identity.subtract(kalmanGain.multiply(measurementMatrix)).multiply(errorCovariance);
    }
}
