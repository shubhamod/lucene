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
package org.apache.lucene.util.hnsw.math.stat.descriptive.rank;

import java.io.Serializable;
import java.util.Arrays;
import java.util.BitSet;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathUnsupportedOperationException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.stat.ranking.NaNStrategy;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.KthSelector;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.MedianOf3PivotingStrategy;
import org.apache.lucene.util.hnsw.math.util.PivotingStrategyInterface;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class Percentile extends AbstractUnivariateStatistic implements Serializable {

    
    private static final long serialVersionUID = -8091216485095130416L;

    
    private static final int MAX_CACHED_LEVELS = 10;

    
    private static final int PIVOTS_HEAP_LENGTH = 0x1 << MAX_CACHED_LEVELS - 1;

    
    private final KthSelector kthSelector;

    
    private final EstimationType estimationType;

    
    private final NaNStrategy nanStrategy;

    
    private double quantile;

    
    private int[] cachedPivots;

    
    public Percentile() {
        // No try-catch or advertised exception here - arg is valid
        this(50.0);
    }

    
    public Percentile(final double quantile) throws MathIllegalArgumentException {
        this(quantile, EstimationType.LEGACY, NaNStrategy.REMOVED,
             new KthSelector(new MedianOf3PivotingStrategy()));
    }

    
    public Percentile(final Percentile original) throws NullArgumentException {

        MathUtils.checkNotNull(original);
        estimationType   = original.getEstimationType();
        nanStrategy      = original.getNaNStrategy();
        kthSelector      = original.getKthSelector();

        setData(original.getDataRef());
        if (original.cachedPivots != null) {
            System.arraycopy(original.cachedPivots, 0, cachedPivots, 0, original.cachedPivots.length);
        }
        setQuantile(original.quantile);

    }

    
    protected Percentile(final double quantile,
                         final EstimationType estimationType,
                         final NaNStrategy nanStrategy,
                         final KthSelector kthSelector)
        throws MathIllegalArgumentException {
        setQuantile(quantile);
        cachedPivots = null;
        MathUtils.checkNotNull(estimationType);
        MathUtils.checkNotNull(nanStrategy);
        MathUtils.checkNotNull(kthSelector);
        this.estimationType = estimationType;
        this.nanStrategy = nanStrategy;
        this.kthSelector = kthSelector;
    }

    
    @Override
    public void setData(final double[] values) {
        if (values == null) {
            cachedPivots = null;
        } else {
            cachedPivots = new int[PIVOTS_HEAP_LENGTH];
            Arrays.fill(cachedPivots, -1);
        }
        super.setData(values);
    }

    
    @Override
    public void setData(final double[] values, final int begin, final int length)
    throws MathIllegalArgumentException {
        if (values == null) {
            cachedPivots = null;
        } else {
            cachedPivots = new int[PIVOTS_HEAP_LENGTH];
            Arrays.fill(cachedPivots, -1);
        }
        super.setData(values, begin, length);
    }

    
    public double evaluate(final double p) throws MathIllegalArgumentException {
        return evaluate(getDataRef(), p);
    }

    
    public double evaluate(final double[] values, final double p)
    throws MathIllegalArgumentException {
        test(values, 0, 0);
        return evaluate(values, 0, values.length, p);
    }

    
    @Override
    public double evaluate(final double[] values, final int start, final int length)
    throws MathIllegalArgumentException {
        return evaluate(values, start, length, quantile);
    }

     
    public double evaluate(final double[] values, final int begin,
                           final int length, final double p)
        throws MathIllegalArgumentException {

        test(values, begin, length);
        if (p > 100 || p <= 0) {
            throw new OutOfRangeException(
                    LocalizedFormats.OUT_OF_BOUNDS_QUANTILE_VALUE, p, 0, 100);
        }
        if (length == 0) {
            return Double.NaN;
        }
        if (length == 1) {
            return values[begin]; // always return single value for n = 1
        }

        final double[] work = getWorkArray(values, begin, length);
        final int[] pivotsHeap = getPivots(values);
        return work.length == 0 ? Double.NaN :
                    estimationType.evaluate(work, pivotsHeap, p, kthSelector);
    }

    
    @Deprecated
    int medianOf3(final double[] work, final int begin, final int end) {
        return new MedianOf3PivotingStrategy().pivotIndex(work, begin, end);
        //throw new MathUnsupportedOperationException();
    }

    
    public double getQuantile() {
        return quantile;
    }

    
    public void setQuantile(final double p) throws MathIllegalArgumentException {
        if (p <= 0 || p > 100) {
            throw new OutOfRangeException(
                    LocalizedFormats.OUT_OF_BOUNDS_QUANTILE_VALUE, p, 0, 100);
        }
        quantile = p;
    }

    
    @Override
    public Percentile copy() {
        return new Percentile(this);
    }

    
    @Deprecated
    public static void copy(final Percentile source, final Percentile dest)
        throws MathUnsupportedOperationException {
        throw new MathUnsupportedOperationException();
    }

    
    protected double[] getWorkArray(final double[] values, final int begin, final int length) {
            final double[] work;
            if (values == getDataRef()) {
                work = getDataRef();
            } else {
                switch (nanStrategy) {
                case MAXIMAL:// Replace NaNs with +INFs
                    work = replaceAndSlice(values, begin, length, Double.NaN, Double.POSITIVE_INFINITY);
                    break;
                case MINIMAL:// Replace NaNs with -INFs
                    work = replaceAndSlice(values, begin, length, Double.NaN, Double.NEGATIVE_INFINITY);
                    break;
                case REMOVED:// Drop NaNs from data
                    work = removeAndSlice(values, begin, length, Double.NaN);
                    break;
                case FAILED:// just throw exception as NaN is un-acceptable
                    work = copyOf(values, begin, length);
                    MathArrays.checkNotNaN(work);
                    break;
                default: //FIXED
                    work = copyOf(values,begin,length);
                    break;
                }
            }
            return work;
    }

    
    private static double[] copyOf(final double[] values, final int begin, final int length) {
        MathArrays.verifyValues(values, begin, length);
        return MathArrays.copyOfRange(values, begin, begin + length);
    }

    
    private static double[] replaceAndSlice(final double[] values,
                                            final int begin, final int length,
                                            final double original,
                                            final double replacement) {
        final double[] temp = copyOf(values, begin, length);
        for(int i = 0; i < length; i++) {
            temp[i] = Precision.equalsIncludingNaN(original, temp[i]) ?
                      replacement : temp[i];
        }
        return temp;
    }

    
    private static double[] removeAndSlice(final double[] values,
                                           final int begin, final int length,
                                           final double removedValue) {
        MathArrays.verifyValues(values, begin, length);
        final double[] temp;
        //BitSet(length) to indicate where the removedValue is located
        final BitSet bits = new BitSet(length);
        for (int i = begin; i < begin+length; i++) {
            if (Precision.equalsIncludingNaN(removedValue, values[i])) {
                bits.set(i - begin);
            }
        }
        //Check if empty then create a new copy
        if (bits.isEmpty()) {
            temp = copyOf(values, begin, length); // Nothing removed, just copy
        } else if(bits.cardinality() == length){
            temp = new double[0];                 // All removed, just empty
        }else {                                   // Some removable, so new
            temp = new double[length - bits.cardinality()];
            int start = begin;  //start index from source array (i.e values)
            int dest = 0;       //dest index in destination array(i.e temp)
            int nextOne = -1;   //nextOne is the index of bit set of next one
            int bitSetPtr = 0;  //bitSetPtr is start index pointer of bitset
            while ((nextOne = bits.nextSetBit(bitSetPtr)) != -1) {
                final int lengthToCopy = nextOne - bitSetPtr;
                System.arraycopy(values, start, temp, dest, lengthToCopy);
                dest += lengthToCopy;
                start = begin + (bitSetPtr = bits.nextClearBit(nextOne));
            }
            //Copy any residue past start index till begin+length
            if (start < begin + length) {
                System.arraycopy(values,start,temp,dest,begin + length - start);
            }
        }
        return temp;
    }

    
    private int[] getPivots(final double[] values) {
        final int[] pivotsHeap;
        if (values == getDataRef()) {
            pivotsHeap = cachedPivots;
        } else {
            pivotsHeap = new int[PIVOTS_HEAP_LENGTH];
            Arrays.fill(pivotsHeap, -1);
        }
        return pivotsHeap;
    }

    
    public EstimationType getEstimationType() {
        return estimationType;
    }

    
    public Percentile withEstimationType(final EstimationType newEstimationType) {
        return new Percentile(quantile, newEstimationType, nanStrategy, kthSelector);
    }

    
    public NaNStrategy getNaNStrategy() {
        return nanStrategy;
    }

    
    public Percentile withNaNStrategy(final NaNStrategy newNaNStrategy) {
        return new Percentile(quantile, estimationType, newNaNStrategy, kthSelector);
    }

    
    public KthSelector getKthSelector() {
        return kthSelector;
    }

    
    public PivotingStrategyInterface getPivotingStrategy() {
        return kthSelector.getPivotingStrategy();
    }

    
    public Percentile withKthSelector(final KthSelector newKthSelector) {
        return new Percentile(quantile, estimationType, nanStrategy,
                                newKthSelector);
    }

    
    public enum EstimationType {
        
        LEGACY("Legacy Apache Commons Math") {
            
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 0d;
                final double maxLimit = 1d;
                return Double.compare(p, minLimit) == 0 ? 0 :
                       Double.compare(p, maxLimit) == 0 ?
                               length : p * (length + 1);
            }
        },
        
        R_1("R-1") {

            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 0d;
                return Double.compare(p, minLimit) == 0 ? 0 : length * p + 0.5;
            }

            
            @Override
            protected double estimate(final double[] values,
                                      final int[] pivotsHeap, final double pos,
                                      final int length, final KthSelector selector) {
                return super.estimate(values, pivotsHeap, FastMath.ceil(pos - 0.5), length, selector);
            }

        },
        
        R_2("R-2") {

            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 0d;
                final double maxLimit = 1d;
                return Double.compare(p, maxLimit) == 0 ? length :
                       Double.compare(p, minLimit) == 0 ? 0 : length * p + 0.5;
            }

            
            @Override
            protected double estimate(final double[] values,
                                      final int[] pivotsHeap, final double pos,
                                      final int length, final KthSelector selector) {
                final double low =
                        super.estimate(values, pivotsHeap, FastMath.ceil(pos - 0.5), length, selector);
                final double high =
                        super.estimate(values, pivotsHeap,FastMath.floor(pos + 0.5), length, selector);
                return (low + high) / 2;
            }

        },
        
        R_3("R-3") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 1d/2 / length;
                return Double.compare(p, minLimit) <= 0 ?
                        0 : FastMath.rint(length * p);
            }

        },
        
        R_4("R-4") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 1d / length;
                final double maxLimit = 1d;
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) == 0 ? length : length * p;
            }

        },
        
        R_5("R-5"){

            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 1d/2 / length;
                final double maxLimit = (length - 0.5) / length;
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) >= 0 ?
                               length : length * p + 0.5;
            }
        },
        
        R_6("R-6"){

            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 1d / (length + 1);
                final double maxLimit = 1d * length / (length + 1);
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) >= 0 ?
                               length : (length + 1) * p;
            }
        },

        
        R_7("R-7") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 0d;
                final double maxLimit = 1d;
                return Double.compare(p, minLimit) == 0 ? 0 :
                       Double.compare(p, maxLimit) == 0 ?
                               length : 1 + (length - 1) * p;
            }

        },

        
        R_8("R-8") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 2 * (1d / 3) / (length + 1d / 3);
                final double maxLimit =
                        (length - 1d / 3) / (length + 1d / 3);
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) >= 0 ? length :
                           (length + 1d / 3) * p + 1d / 3;
            }
        },

        
        R_9("R-9") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 5d/8 / (length + 0.25);
                final double maxLimit = (length - 3d/8) / (length + 0.25);
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) >= 0 ? length :
                               (length + 0.25) * p + 3d/8;
            }

        },
        ;

        
        private final String name;

        
        EstimationType(final String type) {
            this.name = type;
        }

        
        protected abstract double index(final double p, final int length);

        
        protected double estimate(final double[] work, final int[] pivotsHeap,
                                  final double pos, final int length,
                                  final KthSelector selector) {

            final double fpos = FastMath.floor(pos);
            final int intPos = (int) fpos;
            final double dif = pos - fpos;

            if (pos < 1) {
                return selector.select(work, pivotsHeap, 0);
            }
            if (pos >= length) {
                return selector.select(work, pivotsHeap, length - 1);
            }

            final double lower = selector.select(work, pivotsHeap, intPos - 1);
            final double upper = selector.select(work, pivotsHeap, intPos);
            return lower + dif * (upper - lower);
        }

        
        protected double evaluate(final double[] work, final int[] pivotsHeap, final double p,
                                  final KthSelector selector) {
            MathUtils.checkNotNull(work);
            if (p > 100 || p <= 0) {
                throw new OutOfRangeException(LocalizedFormats.OUT_OF_BOUNDS_QUANTILE_VALUE,
                                              p, 0, 100);
            }
            return estimate(work, pivotsHeap, index(p/100d, work.length), work.length, selector);
        }

        
        public double evaluate(final double[] work, final double p, final KthSelector selector) {
            return this.evaluate(work, null, p, selector);
        }

        
        String getName() {
            return name;
        }
    }
}
