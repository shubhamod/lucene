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

import java.util.ArrayList;
import java.util.Collection;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;


class ParameterizedWrapper implements ParameterizedODE {

    
    private final FirstOrderDifferentialEquations fode;

    
    ParameterizedWrapper(final FirstOrderDifferentialEquations ode) {
        this.fode = ode;
    }

    
    public int getDimension() {
        return fode.getDimension();
    }

    
    public void computeDerivatives(double t, double[] y, double[] yDot)
        throws MaxCountExceededException, DimensionMismatchException {
        fode.computeDerivatives(t, y, yDot);
    }

    
    public Collection<String> getParametersNames() {
        return new ArrayList<String>();
    }

    
    public boolean isSupported(String name) {
        return false;
    }

    
    public double getParameter(String name)
        throws UnknownParameterException {
        if (!isSupported(name)) {
            throw new UnknownParameterException(name);
        }
        return Double.NaN;
    }

    
    public void setParameter(String name, double value) {
    }

}
