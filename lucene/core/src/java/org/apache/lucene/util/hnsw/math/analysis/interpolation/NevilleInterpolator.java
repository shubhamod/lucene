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
package org.apache.lucene.util.hnsw.math.analysis.interpolation;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialFunctionLagrangeForm;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;


public class NevilleInterpolator implements UnivariateInterpolator,
    Serializable {

    
    static final long serialVersionUID = 3003707660147873733L;

    
    public PolynomialFunctionLagrangeForm interpolate(double x[], double y[])
        throws DimensionMismatchException,
               NumberIsTooSmallException,
               NonMonotonicSequenceException {
        return new PolynomialFunctionLagrangeForm(x, y);
    }
}
