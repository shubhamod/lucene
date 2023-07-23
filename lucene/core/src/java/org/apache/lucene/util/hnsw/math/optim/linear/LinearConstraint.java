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
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;


public class LinearConstraint implements Serializable {
    
    private static final long serialVersionUID = -764632794033034092L;
    
    private final transient RealVector coefficients;
    
    private final Relationship relationship;
    
    private final double value;

    
    public LinearConstraint(final double[] coefficients,
                            final Relationship relationship,
                            final double value) {
        this(new ArrayRealVector(coefficients), relationship, value);
    }

    
    public LinearConstraint(final RealVector coefficients,
                            final Relationship relationship,
                            final double value) {
        this.coefficients = coefficients;
        this.relationship = relationship;
        this.value        = value;
    }

    
    public LinearConstraint(final double[] lhsCoefficients, final double lhsConstant,
                            final Relationship relationship,
                            final double[] rhsCoefficients, final double rhsConstant) {
        double[] sub = new double[lhsCoefficients.length];
        for (int i = 0; i < sub.length; ++i) {
            sub[i] = lhsCoefficients[i] - rhsCoefficients[i];
        }
        this.coefficients = new ArrayRealVector(sub, false);
        this.relationship = relationship;
        this.value        = rhsConstant - lhsConstant;
    }

    
    public LinearConstraint(final RealVector lhsCoefficients, final double lhsConstant,
                            final Relationship relationship,
                            final RealVector rhsCoefficients, final double rhsConstant) {
        this.coefficients = lhsCoefficients.subtract(rhsCoefficients);
        this.relationship = relationship;
        this.value        = rhsConstant - lhsConstant;
    }

    
    public RealVector getCoefficients() {
        return coefficients;
    }

    
    public Relationship getRelationship() {
        return relationship;
    }

    
    public double getValue() {
        return value;
    }

    
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other instanceof LinearConstraint) {
            LinearConstraint rhs = (LinearConstraint) other;
            return relationship == rhs.relationship &&
                value == rhs.value &&
                coefficients.equals(rhs.coefficients);
        }
        return false;
    }

    
    @Override
    public int hashCode() {
        return relationship.hashCode() ^
            Double.valueOf(value).hashCode() ^
            coefficients.hashCode();
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
