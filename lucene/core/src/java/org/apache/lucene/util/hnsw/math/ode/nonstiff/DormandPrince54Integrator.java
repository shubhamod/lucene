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

package org.apache.lucene.util.hnsw.math.ode.nonstiff;

import org.apache.lucene.util.hnsw.math.util.FastMath;




public class DormandPrince54Integrator extends EmbeddedRungeKuttaIntegrator {

  
  private static final String METHOD_NAME = "Dormand-Prince 5(4)";

  
  private static final double[] STATIC_C = {
    1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0
  };

  
  private static final double[][] STATIC_A = {
    {1.0/5.0},
    {3.0/40.0, 9.0/40.0},
    {44.0/45.0, -56.0/15.0, 32.0/9.0},
    {19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0,  -212.0/729.0},
    {9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0},
    {35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0}
  };

  
  private static final double[] STATIC_B = {
    35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0
  };

  
  private static final double E1 =     71.0 / 57600.0;

  // element 2 is zero, so it is neither stored nor used

  
  private static final double E3 =    -71.0 / 16695.0;

  
  private static final double E4 =     71.0 / 1920.0;

  
  private static final double E5 = -17253.0 / 339200.0;

  
  private static final double E6 =     22.0 / 525.0;

  
  private static final double E7 =     -1.0 / 40.0;

  
  public DormandPrince54Integrator(final double minStep, final double maxStep,
                                   final double scalAbsoluteTolerance,
                                   final double scalRelativeTolerance) {
    super(METHOD_NAME, true, STATIC_C, STATIC_A, STATIC_B, new DormandPrince54StepInterpolator(),
          minStep, maxStep, scalAbsoluteTolerance, scalRelativeTolerance);
  }

  
  public DormandPrince54Integrator(final double minStep, final double maxStep,
                                   final double[] vecAbsoluteTolerance,
                                   final double[] vecRelativeTolerance) {
    super(METHOD_NAME, true, STATIC_C, STATIC_A, STATIC_B, new DormandPrince54StepInterpolator(),
          minStep, maxStep, vecAbsoluteTolerance, vecRelativeTolerance);
  }

  
  @Override
  public int getOrder() {
    return 5;
  }

  
  @Override
  protected double estimateError(final double[][] yDotK,
                                 final double[] y0, final double[] y1,
                                 final double h) {

    double error = 0;

    for (int j = 0; j < mainSetDimension; ++j) {
        final double errSum = E1 * yDotK[0][j] +  E3 * yDotK[2][j] +
                              E4 * yDotK[3][j] +  E5 * yDotK[4][j] +
                              E6 * yDotK[5][j] +  E7 * yDotK[6][j];

        final double yScale = FastMath.max(FastMath.abs(y0[j]), FastMath.abs(y1[j]));
        final double tol = (vecAbsoluteTolerance == null) ?
                           (scalAbsoluteTolerance + scalRelativeTolerance * yScale) :
                               (vecAbsoluteTolerance[j] + vecRelativeTolerance[j] * yScale);
        final double ratio  = h * errSum / tol;
        error += ratio * ratio;

    }

    return FastMath.sqrt(error / mainSetDimension);

  }

}
