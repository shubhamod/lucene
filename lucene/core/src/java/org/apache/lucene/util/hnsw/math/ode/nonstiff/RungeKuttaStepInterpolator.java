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

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

import org.apache.lucene.util.hnsw.math.ode.AbstractIntegrator;
import org.apache.lucene.util.hnsw.math.ode.EquationsMapper;
import org.apache.lucene.util.hnsw.math.ode.sampling.AbstractStepInterpolator;



abstract class RungeKuttaStepInterpolator
  extends AbstractStepInterpolator {

    
    protected double[] previousState;

    
    protected double[][] yDotK;

    
    protected AbstractIntegrator integrator;

  
  protected RungeKuttaStepInterpolator() {
    previousState = null;
    yDotK         = null;
    integrator    = null;
  }

  
  RungeKuttaStepInterpolator(final RungeKuttaStepInterpolator interpolator) {

    super(interpolator);

    if (interpolator.currentState != null) {

      previousState = interpolator.previousState.clone();

      yDotK = new double[interpolator.yDotK.length][];
      for (int k = 0; k < interpolator.yDotK.length; ++k) {
        yDotK[k] = interpolator.yDotK[k].clone();
      }

    } else {
      previousState = null;
      yDotK = null;
    }

    // we cannot keep any reference to the equations in the copy
    // the interpolator should have been finalized before
    integrator = null;

  }

  
  public void reinitialize(final AbstractIntegrator rkIntegrator,
                           final double[] y, final double[][] yDotArray, final boolean forward,
                           final EquationsMapper primaryMapper,
                           final EquationsMapper[] secondaryMappers) {
    reinitialize(y, forward, primaryMapper, secondaryMappers);
    this.previousState = null;
    this.yDotK = yDotArray;
    this.integrator = rkIntegrator;
  }

  
  @Override
  public void shift() {
    previousState = currentState.clone();
    super.shift();
  }

  
  @Override
  public void writeExternal(final ObjectOutput out)
    throws IOException {

    // save the state of the base class
    writeBaseExternal(out);

    // save the local attributes
    final int n = (currentState == null) ? -1 : currentState.length;
    for (int i = 0; i < n; ++i) {
      out.writeDouble(previousState[i]);
    }

    final int kMax = (yDotK == null) ? -1 : yDotK.length;
    out.writeInt(kMax);
    for (int k = 0; k < kMax; ++k) {
      for (int i = 0; i < n; ++i) {
        out.writeDouble(yDotK[k][i]);
      }
    }

    // we do not save any reference to the equations

  }

  
  @Override
  public void readExternal(final ObjectInput in)
    throws IOException, ClassNotFoundException {

    // read the base class
    final double t = readBaseExternal(in);

    // read the local attributes
    final int n = (currentState == null) ? -1 : currentState.length;
    if (n < 0) {
      previousState = null;
    } else {
      previousState = new double[n];
      for (int i = 0; i < n; ++i) {
        previousState[i] = in.readDouble();
      }
    }

    final int kMax = in.readInt();
    yDotK = (kMax < 0) ? null : new double[kMax][];
    for (int k = 0; k < kMax; ++k) {
      yDotK[k] = (n < 0) ? null : new double[n];
      for (int i = 0; i < n; ++i) {
        yDotK[k][i] = in.readDouble();
      }
    }

    integrator = null;

    if (currentState != null) {
        // we can now set the interpolated time and state
        setInterpolatedTime(t);
    } else {
        interpolatedTime = t;
    }

  }

}
