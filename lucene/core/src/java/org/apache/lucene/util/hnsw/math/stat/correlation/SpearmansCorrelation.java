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

package org.apache.lucene.util.hnsw.math.stat.correlation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.linear.BlockRealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.stat.ranking.NaNStrategy;
import org.apache.lucene.util.hnsw.math.stat.ranking.NaturalRanking;
import org.apache.lucene.util.hnsw.math.stat.ranking.RankingAlgorithm;


public class SpearmansCorrelation {

    
    private final RealMatrix data;

    
    private final RankingAlgorithm rankingAlgorithm;

    
    private final PearsonsCorrelation rankCorrelation;

    
    public SpearmansCorrelation() {
        this(new NaturalRanking());
    }

    
    public SpearmansCorrelation(final RankingAlgorithm rankingAlgorithm) {
        data = null;
        this.rankingAlgorithm = rankingAlgorithm;
        rankCorrelation = null;
    }

    
    public SpearmansCorrelation(final RealMatrix dataMatrix) {
        this(dataMatrix, new NaturalRanking());
    }

    
    public SpearmansCorrelation(final RealMatrix dataMatrix, final RankingAlgorithm rankingAlgorithm) {
        this.rankingAlgorithm = rankingAlgorithm;
        this.data = rankTransform(dataMatrix);
        rankCorrelation = new PearsonsCorrelation(data);
    }

    
    public RealMatrix getCorrelationMatrix() {
        return rankCorrelation.getCorrelationMatrix();
    }

    
    public PearsonsCorrelation getRankCorrelation() {
        return rankCorrelation;
    }

    
    public RealMatrix computeCorrelationMatrix(final RealMatrix matrix) {
        final RealMatrix matrixCopy = rankTransform(matrix);
        return new PearsonsCorrelation().computeCorrelationMatrix(matrixCopy);
    }

    
    public RealMatrix computeCorrelationMatrix(final double[][] matrix) {
       return computeCorrelationMatrix(new BlockRealMatrix(matrix));
    }

    
    public double correlation(final double[] xArray, final double[] yArray) {
        if (xArray.length != yArray.length) {
            throw new DimensionMismatchException(xArray.length, yArray.length);
        } else if (xArray.length < 2) {
            throw new MathIllegalArgumentException(LocalizedFormats.INSUFFICIENT_DIMENSION,
                                                   xArray.length, 2);
        } else {
            double[] x = xArray;
            double[] y = yArray;
            if (rankingAlgorithm instanceof NaturalRanking &&
                NaNStrategy.REMOVED == ((NaturalRanking) rankingAlgorithm).getNanStrategy()) {
                final Set<Integer> nanPositions = new HashSet<Integer>();

                nanPositions.addAll(getNaNPositions(xArray));
                nanPositions.addAll(getNaNPositions(yArray));

                x = removeValues(xArray, nanPositions);
                y = removeValues(yArray, nanPositions);
            }
            return new PearsonsCorrelation().correlation(rankingAlgorithm.rank(x), rankingAlgorithm.rank(y));
        }
    }

    
    private RealMatrix rankTransform(final RealMatrix matrix) {
        RealMatrix transformed = null;

        if (rankingAlgorithm instanceof NaturalRanking &&
                ((NaturalRanking) rankingAlgorithm).getNanStrategy() == NaNStrategy.REMOVED) {
            final Set<Integer> nanPositions = new HashSet<Integer>();
            for (int i = 0; i < matrix.getColumnDimension(); i++) {
                nanPositions.addAll(getNaNPositions(matrix.getColumn(i)));
            }

            // if we have found NaN values, we have to update the matrix size
            if (!nanPositions.isEmpty()) {
                transformed = new BlockRealMatrix(matrix.getRowDimension() - nanPositions.size(),
                                                  matrix.getColumnDimension());
                for (int i = 0; i < transformed.getColumnDimension(); i++) {
                    transformed.setColumn(i, removeValues(matrix.getColumn(i), nanPositions));
                }
            }
        }

        if (transformed == null) {
            transformed = matrix.copy();
        }

        for (int i = 0; i < transformed.getColumnDimension(); i++) {
            transformed.setColumn(i, rankingAlgorithm.rank(transformed.getColumn(i)));
        }

        return transformed;
    }

    
    private List<Integer> getNaNPositions(final double[] input) {
        final List<Integer> positions = new ArrayList<Integer>();
        for (int i = 0; i < input.length; i++) {
            if (Double.isNaN(input[i])) {
                positions.add(i);
            }
        }
        return positions;
    }

    
    private double[] removeValues(final double[] input, final Set<Integer> indices) {
        if (indices.isEmpty()) {
            return input;
        }
        final double[] result = new double[input.length - indices.size()];
        for (int i = 0, j = 0; i < input.length; i++) {
            if (!indices.contains(i)) {
                result[j++] = input[i];
            }
        }
        return result;
    }
}
