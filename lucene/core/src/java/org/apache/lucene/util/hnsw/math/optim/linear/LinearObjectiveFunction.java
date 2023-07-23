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
package org.apache.lucene.util.hnsw.math.optim.linear;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import org.apache.lucene.util.hnsw.math.analysis.MultivariateFunction;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.optim.OptimizationData;


public class LinearObjectiveFunction
    implements MultivariateFunction,
               OptimizationData,
               Serializable {
    
    private static final long serialVersionUID = -4531815507568396090L;
    
    private final transient RealVector coefficients;
    
    private final double constantTerm;

    
    public LinearObjectiveFunction(double[] coefficients, double constantTerm) {
        this(new ArrayRealVector(coefficients), constantTerm);
    }

    
    public LinearObjectiveFunction(RealVector coefficients, double constantTerm) {
        this.coefficients = coefficients;
        this.constantTerm = constantTerm;
    }

    
    public RealVector getCoefficients() {
        return coefficients;
    }

    
    public double getConstantTerm() {
        return constantTerm;
    }

    
    public double value(final double[] point) {
        return value(new ArrayRealVector(point, false));
    }

    
    public double value(final RealVector point) {
        return coefficients.dotProduct(point) + constantTerm;
    }

    
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other instanceof LinearObjectiveFunction) {
            LinearObjectiveFunction rhs = (LinearObjectiveFunction) other;
          return (constantTerm == rhs.constantTerm) && coefficients.equals(rhs.coefficients);
        }

        return false;
    }

    
    @Override
    public int hashCode() {
        return Double.valueOf(constantTerm).hashCode() ^ coefficients.hashCode();
    }

    
    private void writeObject(ObjectOutputStream oos)
        throws IOException {
        oos.defaultWriteObject();
        MatrixUtils.serializeRealVector(coefficients, oos);
    }

    
    private void readObject(ObjectInputStream ois)
      throws ClassNotFoundException, IOException {
        ois.defaultReadObject();
        MatrixUtils.deserializeRealVector(this, "coefficients", ois);
    }
}
