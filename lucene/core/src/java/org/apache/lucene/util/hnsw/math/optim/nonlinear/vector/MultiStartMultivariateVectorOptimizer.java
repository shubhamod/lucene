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
package org.apache.lucene.util.hnsw.math.optim.nonlinear.vector;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.random.RandomVectorGenerator;
import org.apache.lucene.util.hnsw.math.optim.BaseMultiStartMultivariateOptimizer;
import org.apache.lucene.util.hnsw.math.optim.PointVectorValuePair;


@Deprecated
public class MultiStartMultivariateVectorOptimizer
    extends BaseMultiStartMultivariateOptimizer<PointVectorValuePair> {
    
    private final MultivariateVectorOptimizer optimizer;
    
    private final List<PointVectorValuePair> optima = new ArrayList<PointVectorValuePair>();

    
    public MultiStartMultivariateVectorOptimizer(final MultivariateVectorOptimizer optimizer,
                                                 final int starts,
                                                 final RandomVectorGenerator generator)
        throws NullArgumentException,
        NotStrictlyPositiveException {
        super(optimizer, starts, generator);
        this.optimizer = optimizer;
    }

    
    @Override
    public PointVectorValuePair[] getOptima() {
        Collections.sort(optima, getPairComparator());
        return optima.toArray(new PointVectorValuePair[0]);
    }

    
    @Override
    protected void store(PointVectorValuePair optimum) {
        optima.add(optimum);
    }

    
    @Override
    protected void clear() {
        optima.clear();
    }

    
    private Comparator<PointVectorValuePair> getPairComparator() {
        return new Comparator<PointVectorValuePair>() {
            
            private final RealVector target = new ArrayRealVector(optimizer.getTarget(), false);
            
            private final RealMatrix weight = optimizer.getWeight();

            
            public int compare(final PointVectorValuePair o1,
                               final PointVectorValuePair o2) {
                if (o1 == null) {
                    return (o2 == null) ? 0 : 1;
                } else if (o2 == null) {
                    return -1;
                }
                return Double.compare(weightedResidual(o1),
                                      weightedResidual(o2));
            }

            private double weightedResidual(final PointVectorValuePair pv) {
                final RealVector v = new ArrayRealVector(pv.getValueRef(), false);
                final RealVector r = target.subtract(v);
                return r.dotProduct(weight.operate(r));
            }
        };
    }
}
