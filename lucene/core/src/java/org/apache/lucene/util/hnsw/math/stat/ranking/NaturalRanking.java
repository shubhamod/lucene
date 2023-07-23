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

package org.apache.lucene.util.hnsw.math.stat.ranking;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.NotANumberException;
import org.apache.lucene.util.hnsw.math.random.RandomDataGenerator;
import org.apache.lucene.util.hnsw.math.random.RandomGenerator;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class NaturalRanking implements RankingAlgorithm {

    
    public static final NaNStrategy DEFAULT_NAN_STRATEGY = NaNStrategy.FAILED;

    
    public static final TiesStrategy DEFAULT_TIES_STRATEGY = TiesStrategy.AVERAGE;

    
    private final NaNStrategy nanStrategy;

    
    private final TiesStrategy tiesStrategy;

    
    private final RandomDataGenerator randomData;

    
    public NaturalRanking() {
        super();
        tiesStrategy = DEFAULT_TIES_STRATEGY;
        nanStrategy = DEFAULT_NAN_STRATEGY;
        randomData = null;
    }

    
    public NaturalRanking(TiesStrategy tiesStrategy) {
        super();
        this.tiesStrategy = tiesStrategy;
        nanStrategy = DEFAULT_NAN_STRATEGY;
        randomData = new RandomDataGenerator();
    }

    
    public NaturalRanking(NaNStrategy nanStrategy) {
        super();
        this.nanStrategy = nanStrategy;
        tiesStrategy = DEFAULT_TIES_STRATEGY;
        randomData = null;
    }

    
    public NaturalRanking(NaNStrategy nanStrategy, TiesStrategy tiesStrategy) {
        super();
        this.nanStrategy = nanStrategy;
        this.tiesStrategy = tiesStrategy;
        randomData = new RandomDataGenerator();
    }

    
    public NaturalRanking(RandomGenerator randomGenerator) {
        super();
        this.tiesStrategy = TiesStrategy.RANDOM;
        nanStrategy = DEFAULT_NAN_STRATEGY;
        randomData = new RandomDataGenerator(randomGenerator);
    }


    
    public NaturalRanking(NaNStrategy nanStrategy,
            RandomGenerator randomGenerator) {
        super();
        this.nanStrategy = nanStrategy;
        this.tiesStrategy = TiesStrategy.RANDOM;
        randomData = new RandomDataGenerator(randomGenerator);
    }

    
    public NaNStrategy getNanStrategy() {
        return nanStrategy;
    }

    
    public TiesStrategy getTiesStrategy() {
        return tiesStrategy;
    }

    
    public double[] rank(double[] data) {

        // Array recording initial positions of data to be ranked
        IntDoublePair[] ranks = new IntDoublePair[data.length];
        for (int i = 0; i < data.length; i++) {
            ranks[i] = new IntDoublePair(data[i], i);
        }

        // Recode, remove or record positions of NaNs
        List<Integer> nanPositions = null;
        switch (nanStrategy) {
            case MAXIMAL: // Replace NaNs with +INFs
                recodeNaNs(ranks, Double.POSITIVE_INFINITY);
                break;
            case MINIMAL: // Replace NaNs with -INFs
                recodeNaNs(ranks, Double.NEGATIVE_INFINITY);
                break;
            case REMOVED: // Drop NaNs from data
                ranks = removeNaNs(ranks);
                break;
            case FIXED:   // Record positions of NaNs
                nanPositions = getNanPositions(ranks);
                break;
            case FAILED:
                nanPositions = getNanPositions(ranks);
                if (nanPositions.size() > 0) {
                    throw new NotANumberException();
                }
                break;
            default: // this should not happen unless NaNStrategy enum is changed
                throw new MathInternalError();
        }

        // Sort the IntDoublePairs
        Arrays.sort(ranks);

        // Walk the sorted array, filling output array using sorted positions,
        // resolving ties as we go
        double[] out = new double[ranks.length];
        int pos = 1;  // position in sorted array
        out[ranks[0].getPosition()] = pos;
        List<Integer> tiesTrace = new ArrayList<Integer>();
        tiesTrace.add(ranks[0].getPosition());
        for (int i = 1; i < ranks.length; i++) {
            if (Double.compare(ranks[i].getValue(), ranks[i - 1].getValue()) > 0) {
                // tie sequence has ended (or had length 1)
                pos = i + 1;
                if (tiesTrace.size() > 1) {  // if seq is nontrivial, resolve
                    resolveTie(out, tiesTrace);
                }
                tiesTrace = new ArrayList<Integer>();
                tiesTrace.add(ranks[i].getPosition());
            } else {
                // tie sequence continues
                tiesTrace.add(ranks[i].getPosition());
            }
            out[ranks[i].getPosition()] = pos;
        }
        if (tiesTrace.size() > 1) {  // handle tie sequence at end
            resolveTie(out, tiesTrace);
        }
        if (nanStrategy == NaNStrategy.FIXED) {
            restoreNaNs(out, nanPositions);
        }
        return out;
    }

    
    private IntDoublePair[] removeNaNs(IntDoublePair[] ranks) {
        if (!containsNaNs(ranks)) {
            return ranks;
        }
        IntDoublePair[] outRanks = new IntDoublePair[ranks.length];
        int j = 0;
        for (int i = 0; i < ranks.length; i++) {
            if (Double.isNaN(ranks[i].getValue())) {
                // drop, but adjust original ranks of later elements
                for (int k = i + 1; k < ranks.length; k++) {
                    ranks[k] = new IntDoublePair(
                            ranks[k].getValue(), ranks[k].getPosition() - 1);
                }
            } else {
                outRanks[j] = new IntDoublePair(
                        ranks[i].getValue(), ranks[i].getPosition());
                j++;
            }
        }
        IntDoublePair[] returnRanks = new IntDoublePair[j];
        System.arraycopy(outRanks, 0, returnRanks, 0, j);
        return returnRanks;
    }

    
    private void recodeNaNs(IntDoublePair[] ranks, double value) {
        for (int i = 0; i < ranks.length; i++) {
            if (Double.isNaN(ranks[i].getValue())) {
                ranks[i] = new IntDoublePair(
                        value, ranks[i].getPosition());
            }
        }
    }

    
    private boolean containsNaNs(IntDoublePair[] ranks) {
        for (int i = 0; i < ranks.length; i++) {
            if (Double.isNaN(ranks[i].getValue())) {
                return true;
            }
        }
        return false;
    }

    
    private void resolveTie(double[] ranks, List<Integer> tiesTrace) {

        // constant value of ranks over tiesTrace
        final double c = ranks[tiesTrace.get(0)];

        // length of sequence of tied ranks
        final int length = tiesTrace.size();

        switch (tiesStrategy) {
            case  AVERAGE:  // Replace ranks with average
                fill(ranks, tiesTrace, (2 * c + length - 1) / 2d);
                break;
            case MAXIMUM:   // Replace ranks with maximum values
                fill(ranks, tiesTrace, c + length - 1);
                break;
            case MINIMUM:   // Replace ties with minimum
                fill(ranks, tiesTrace, c);
                break;
            case RANDOM:    // Fill with random integral values in [c, c + length - 1]
                Iterator<Integer> iterator = tiesTrace.iterator();
                long f = FastMath.round(c);
                while (iterator.hasNext()) {
                    // No advertised exception because args are guaranteed valid
                    ranks[iterator.next()] =
                        randomData.nextLong(f, f + length - 1);
                }
                break;
            case SEQUENTIAL:  // Fill sequentially from c to c + length - 1
                // walk and fill
                iterator = tiesTrace.iterator();
                f = FastMath.round(c);
                int i = 0;
                while (iterator.hasNext()) {
                    ranks[iterator.next()] = f + i++;
                }
                break;
            default: // this should not happen unless TiesStrategy enum is changed
                throw new MathInternalError();
        }
    }

    
    private void fill(double[] data, List<Integer> tiesTrace, double value) {
        Iterator<Integer> iterator = tiesTrace.iterator();
        while (iterator.hasNext()) {
            data[iterator.next()] = value;
        }
    }

    
    private void restoreNaNs(double[] ranks, List<Integer> nanPositions) {
        if (nanPositions.size() == 0) {
            return;
        }
        Iterator<Integer> iterator = nanPositions.iterator();
        while (iterator.hasNext()) {
            ranks[iterator.next().intValue()] = Double.NaN;
        }

    }

    
    private List<Integer> getNanPositions(IntDoublePair[] ranks) {
        ArrayList<Integer> out = new ArrayList<Integer>();
        for (int i = 0; i < ranks.length; i++) {
            if (Double.isNaN(ranks[i].getValue())) {
                out.add(Integer.valueOf(i));
            }
        }
        return out;
    }

    
    private static class IntDoublePair implements Comparable<IntDoublePair>  {

        
        private final double value;

        
        private final int position;

        
        IntDoublePair(double value, int position) {
            this.value = value;
            this.position = position;
        }

        
        public int compareTo(IntDoublePair other) {
            return Double.compare(value, other.value);
        }

        // N.B. equals() and hashCode() are not implemented; see MATH-610 for discussion.

        
        public double getValue() {
            return value;
        }

        
        public int getPosition() {
            return position;
        }
    }
}
