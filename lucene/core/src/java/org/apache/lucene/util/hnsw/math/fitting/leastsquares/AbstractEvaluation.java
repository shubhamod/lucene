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
package org.apache.lucene.util.hnsw.math.fitting.leastsquares;

import org.apache.lucene.util.hnsw.math.fitting.leastsquares.LeastSquaresProblem.Evaluation;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.DecompositionSolver;
import org.apache.lucene.util.hnsw.math.linear.QRDecomposition;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public abstract class AbstractEvaluation implements Evaluation {

    
    private final int observationSize;

    
    AbstractEvaluation(final int observationSize) {
        this.observationSize = observationSize;
    }

    
    public RealMatrix getCovariances(double threshold) {
        // Set up the Jacobian.
        final RealMatrix j = this.getJacobian();

        // Compute transpose(J)J.
        final RealMatrix jTj = j.transpose().multiply(j);

        // Compute the covariances matrix.
        final DecompositionSolver solver
                = new QRDecomposition(jTj, threshold).getSolver();
        return solver.getInverse();
    }

    
    public RealVector getSigma(double covarianceSingularityThreshold) {
        final RealMatrix cov = this.getCovariances(covarianceSingularityThreshold);
        final int nC = cov.getColumnDimension();
        final RealVector sig = new ArrayRealVector(nC);
        for (int i = 0; i < nC; ++i) {
            sig.setEntry(i, FastMath.sqrt(cov.getEntry(i,i)));
        }
        return sig;
    }

    
    public double getRMS() {
        final double cost = this.getCost();
        return FastMath.sqrt(cost * cost / this.observationSize);
    }

    
    public double getCost() {
        final ArrayRealVector r = new ArrayRealVector(this.getResiduals());
        return FastMath.sqrt(r.dotProduct(r));
    }

}
