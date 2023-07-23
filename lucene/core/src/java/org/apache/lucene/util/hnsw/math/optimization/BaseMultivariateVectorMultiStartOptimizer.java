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

package org.apache.lucene.util.hnsw.math.optimization;

import java.util.Arrays;
import java.util.Comparator;

import org.apache.lucene.util.hnsw.math.analysis.MultivariateVectorFunction;
import org.apache.lucene.util.hnsw.math.exception.ConvergenceException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.random.RandomVectorGenerator;


@Deprecated
public class BaseMultivariateVectorMultiStartOptimizer<FUNC extends MultivariateVectorFunction>
    implements BaseMultivariateVectorOptimizer<FUNC> {
    
    private final BaseMultivariateVectorOptimizer<FUNC> optimizer;
    
    private int maxEvaluations;
    
    private int totalEvaluations;
    
    private int starts;
    
    private RandomVectorGenerator generator;
    
    private PointVectorValuePair[] optima;

    
    protected BaseMultivariateVectorMultiStartOptimizer(final BaseMultivariateVectorOptimizer<FUNC> optimizer,
                                                           final int starts,
                                                           final RandomVectorGenerator generator) {
        if (optimizer == null ||
            generator == null) {
            throw new NullArgumentException();
        }
        if (starts < 1) {
            throw new NotStrictlyPositiveException(starts);
        }

        this.optimizer = optimizer;
        this.starts = starts;
        this.generator = generator;
    }

    
    public PointVectorValuePair[] getOptima() {
        if (optima == null) {
            throw new MathIllegalStateException(LocalizedFormats.NO_OPTIMUM_COMPUTED_YET);
        }
        return optima.clone();
    }

    
    public int getMaxEvaluations() {
        return maxEvaluations;
    }

    
    public int getEvaluations() {
        return totalEvaluations;
    }

    
    public ConvergenceChecker<PointVectorValuePair> getConvergenceChecker() {
        return optimizer.getConvergenceChecker();
    }

    
    public PointVectorValuePair optimize(int maxEval, final FUNC f,
                                            double[] target, double[] weights,
                                            double[] startPoint) {
        maxEvaluations = maxEval;
        RuntimeException lastException = null;
        optima = new PointVectorValuePair[starts];
        totalEvaluations = 0;

        // Multi-start loop.
        for (int i = 0; i < starts; ++i) {

            // CHECKSTYLE: stop IllegalCatch
            try {
                optima[i] = optimizer.optimize(maxEval - totalEvaluations, f, target, weights,
                                               i == 0 ? startPoint : generator.nextVector());
            } catch (ConvergenceException oe) {
                optima[i] = null;
            } catch (RuntimeException mue) {
                lastException = mue;
                optima[i] = null;
            }
            // CHECKSTYLE: resume IllegalCatch

            totalEvaluations += optimizer.getEvaluations();
        }

        sortPairs(target, weights);

        if (optima[0] == null) {
            throw lastException; // cannot be null if starts >=1
        }

        // Return the found point given the best objective function value.
        return optima[0];
    }

    
    private void sortPairs(final double[] target,
                           final double[] weights) {
        Arrays.sort(optima, new Comparator<PointVectorValuePair>() {
                
                public int compare(final PointVectorValuePair o1,
                                   final PointVectorValuePair o2) {
                    if (o1 == null) {
                        return (o2 == null) ? 0 : 1;
                    } else if (o2 == null) {
                        return -1;
                    }
                    return Double.compare(weightedResidual(o1), weightedResidual(o2));
                }
                private double weightedResidual(final PointVectorValuePair pv) {
                    final double[] value = pv.getValueRef();
                    double sum = 0;
                    for (int i = 0; i < value.length; ++i) {
                        final double ri = value[i] - target[i];
                        sum += weights[i] * ri * ri;
                    }
                    return sum;
                }
            });
    }
}
