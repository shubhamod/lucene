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
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;


class DenseWeightedEvaluation extends AbstractEvaluation {

    
    private final Evaluation unweighted;
    
    private final RealMatrix weightSqrt;

    
    DenseWeightedEvaluation(final Evaluation unweighted,
                            final RealMatrix weightSqrt) {
        // weight square root is square, nR=nC=number of observations
        super(weightSqrt.getColumnDimension());
        this.unweighted = unweighted;
        this.weightSqrt = weightSqrt;
    }

    /* apply weights */

    
    public RealMatrix getJacobian() {
        return weightSqrt.multiply(this.unweighted.getJacobian());
    }

    
    public RealVector getResiduals() {
        return this.weightSqrt.operate(this.unweighted.getResiduals());
    }

    /* delegate */

    
    public RealVector getPoint() {
        return unweighted.getPoint();
    }

}
