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
import org.apache.lucene.util.hnsw.math.analysis.polynomials.PolynomialFunctionNewtonForm;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.NonMonotonicSequenceException;


public class DividedDifferenceInterpolator
    implements UnivariateInterpolator, Serializable {
    
    private static final long serialVersionUID = 107049519551235069L;

    
    public PolynomialFunctionNewtonForm interpolate(double x[], double y[])
        throws DimensionMismatchException,
               NumberIsTooSmallException,
               NonMonotonicSequenceException {
        
        PolynomialFunctionLagrangeForm.verifyInterpolationArray(x, y, true);

        
        final double[] c = new double[x.length-1];
        System.arraycopy(x, 0, c, 0, c.length);

        final double[] a = computeDividedDifference(x, y);
        return new PolynomialFunctionNewtonForm(a, c);
    }

    
    protected static double[] computeDividedDifference(final double x[], final double y[])
        throws DimensionMismatchException,
               NumberIsTooSmallException,
               NonMonotonicSequenceException {
        PolynomialFunctionLagrangeForm.verifyInterpolationArray(x, y, true);

        final double[] divdiff = y.clone(); // initialization

        final int n = x.length;
        final double[] a = new double [n];
        a[0] = divdiff[0];
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < n-i; j++) {
                final double denominator = x[j+i] - x[j];
                divdiff[j] = (divdiff[j+1] - divdiff[j]) / denominator;
            }
            a[i] = divdiff[0];
        }

        return a;
    }
}
