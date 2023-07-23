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

package org.apache.lucene.util.hnsw.math.ode;




public class FirstOrderConverter implements FirstOrderDifferentialEquations {

    
    private final SecondOrderDifferentialEquations equations;

    
    private final int dimension;

    
    private final double[] z;

    
    private final double[] zDot;

    
    private final double[] zDDot;

  
  public FirstOrderConverter (final SecondOrderDifferentialEquations equations) {
      this.equations = equations;
      dimension      = equations.getDimension();
      z              = new double[dimension];
      zDot           = new double[dimension];
      zDDot          = new double[dimension];
  }

  
  public int getDimension() {
    return 2 * dimension;
  }

  
  public void computeDerivatives(final double t, final double[] y, final double[] yDot) {

    // split the state vector in two
    System.arraycopy(y, 0,         z,    0, dimension);
    System.arraycopy(y, dimension, zDot, 0, dimension);

    // apply the underlying equations set
    equations.computeSecondDerivatives(t, z, zDot, zDDot);

    // build the result state derivative
    System.arraycopy(zDot,  0, yDot, 0,         dimension);
    System.arraycopy(zDDot, 0, yDot, dimension, dimension);

  }

}
