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

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;


class ParameterJacobianWrapper implements ParameterJacobianProvider {

    
    private final FirstOrderDifferentialEquations fode;

    
    private final ParameterizedODE pode;

    
    private final Map<String, Double> hParam;

    
    ParameterJacobianWrapper(final FirstOrderDifferentialEquations fode,
                             final ParameterizedODE pode,
                             final ParameterConfiguration[] paramsAndSteps) {
        this.fode = fode;
        this.pode = pode;
        this.hParam = new HashMap<String, Double>();

        // set up parameters for jacobian computation
        for (final ParameterConfiguration param : paramsAndSteps) {
            final String name = param.getParameterName();
            if (pode.isSupported(name)) {
                hParam.put(name, param.getHP());
            }
        }
    }

    
    public Collection<String> getParametersNames() {
        return pode.getParametersNames();
    }

    
    public boolean isSupported(String name) {
        return pode.isSupported(name);
    }

    
    public void computeParameterJacobian(double t, double[] y, double[] yDot,
                                         String paramName, double[] dFdP)
        throws DimensionMismatchException, MaxCountExceededException {

        final int n = fode.getDimension();
        if (pode.isSupported(paramName)) {
            final double[] tmpDot = new double[n];

            // compute the jacobian df/dp w.r.t. parameter
            final double p  = pode.getParameter(paramName);
            final double hP = hParam.get(paramName);
            pode.setParameter(paramName, p + hP);
            fode.computeDerivatives(t, y, tmpDot);
            for (int i = 0; i < n; ++i) {
                dFdP[i] = (tmpDot[i] - yDot[i]) / hP;
            }
            pode.setParameter(paramName, p);
        } else {
            Arrays.fill(dFdP, 0, n, 0.0);
        }

    }

}
