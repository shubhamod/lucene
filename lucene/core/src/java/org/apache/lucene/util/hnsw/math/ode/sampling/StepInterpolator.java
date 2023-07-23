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

package org.apache.lucene.util.hnsw.math.ode.sampling;

import java.io.Externalizable;

import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;



public interface StepInterpolator extends Externalizable {

  
  double getPreviousTime();

  
  double getCurrentTime();

  
  double getInterpolatedTime();

  
  void setInterpolatedTime(double time);

  
  double[] getInterpolatedState() throws MaxCountExceededException;

  
  double[] getInterpolatedDerivatives() throws MaxCountExceededException;

  
  double[] getInterpolatedSecondaryState(int index) throws MaxCountExceededException;

  
  double[] getInterpolatedSecondaryDerivatives(int index) throws MaxCountExceededException;

  
  boolean isForward();

  
   StepInterpolator copy() throws MaxCountExceededException;

}
