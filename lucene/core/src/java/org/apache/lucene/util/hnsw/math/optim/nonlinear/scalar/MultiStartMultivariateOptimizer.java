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
package org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.random.RandomVectorGenerator;
import org.apache.lucene.util.hnsw.math.optim.BaseMultiStartMultivariateOptimizer;
import org.apache.lucene.util.hnsw.math.optim.PointValuePair;


public class MultiStartMultivariateOptimizer
    extends BaseMultiStartMultivariateOptimizer<PointValuePair> {
    
    private final MultivariateOptimizer optimizer;
    
    private final List<PointValuePair> optima = new ArrayList<PointValuePair>();

    
    public MultiStartMultivariateOptimizer(final MultivariateOptimizer optimizer,
                                           final int starts,
                                           final RandomVectorGenerator generator)
        throws NullArgumentException,
        NotStrictlyPositiveException {
        super(optimizer, starts, generator);
        this.optimizer = optimizer;
    }

    
    @Override
    public PointValuePair[] getOptima() {
        Collections.sort(optima, getPairComparator());
        return optima.toArray(new PointValuePair[0]);
    }

    
    @Override
    protected void store(PointValuePair optimum) {
        optima.add(optimum);
    }

    
    @Override
    protected void clear() {
        optima.clear();
    }

    
    private Comparator<PointValuePair> getPairComparator() {
        return new Comparator<PointValuePair>() {
            
            public int compare(final PointValuePair o1,
                               final PointValuePair o2) {
                if (o1 == null) {
                    return (o2 == null) ? 0 : 1;
                } else if (o2 == null) {
                    return -1;
                }
                final double v1 = o1.getValue();
                final double v2 = o2.getValue();
                return (optimizer.getGoalType() == GoalType.MINIMIZE) ?
                    Double.compare(v1, v2) : Double.compare(v2, v1);
            }
        };
    }
}
