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

package org.apache.lucene.util.hnsw.math.random;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.util.hnsw.math.distribution.AbstractRealDistribution;
import org.apache.lucene.util.hnsw.math.distribution.ConstantRealDistribution;
import org.apache.lucene.util.hnsw.math.distribution.NormalDistribution;
import org.apache.lucene.util.hnsw.math.distribution.RealDistribution;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.stat.descriptive.StatisticalSummary;
import org.apache.lucene.util.hnsw.math.stat.descriptive.SummaryStatistics;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class EmpiricalDistribution extends AbstractRealDistribution {

    
    public static final int DEFAULT_BIN_COUNT = 1000;

    
    private static final String FILE_CHARSET = "US-ASCII";

    
    private static final long serialVersionUID = 5729073523949762654L;

    
    protected final RandomDataGenerator randomData;

    
    private final List<SummaryStatistics> binStats;

    
    private SummaryStatistics sampleStats = null;

    
    private double max = Double.NEGATIVE_INFINITY;

    
    private double min = Double.POSITIVE_INFINITY;

    
    private double delta = 0d;

    
    private final int binCount;

    
    private boolean loaded = false;

    
    private double[] upperBounds = null;

    
    public EmpiricalDistribution() {
        this(DEFAULT_BIN_COUNT);
    }

    
    public EmpiricalDistribution(int binCount) {
        this(binCount, new RandomDataGenerator());
    }

    
    public EmpiricalDistribution(int binCount, RandomGenerator generator) {
        this(binCount, new RandomDataGenerator(generator));
    }

    
    public EmpiricalDistribution(RandomGenerator generator) {
        this(DEFAULT_BIN_COUNT, generator);
    }

    
    @Deprecated
    public EmpiricalDistribution(int binCount, RandomDataImpl randomData) {
        this(binCount, randomData.getDelegate());
    }

    
    @Deprecated
    public EmpiricalDistribution(RandomDataImpl randomData) {
        this(DEFAULT_BIN_COUNT, randomData);
    }

    
    private EmpiricalDistribution(int binCount,
                                  RandomDataGenerator randomData) {
        super(randomData.getRandomGenerator());
        if (binCount <= 0) {
            throw new NotStrictlyPositiveException(binCount);
        }
        this.binCount = binCount;
        this.randomData = randomData;
        binStats = new ArrayList<SummaryStatistics>();
    }

    
    public void load(double[] in) throws NullArgumentException {
        DataAdapter da = new ArrayDataAdapter(in);
        try {
            da.computeStats();
            // new adapter for the second pass
            fillBinStats(new ArrayDataAdapter(in));
        } catch (IOException ex) {
            // Can't happen
            throw new MathInternalError();
        }
        loaded = true;

    }

    
    public void load(URL url) throws IOException, NullArgumentException, ZeroException {
        MathUtils.checkNotNull(url);
        Charset charset = Charset.forName(FILE_CHARSET);
        BufferedReader in =
            new BufferedReader(new InputStreamReader(url.openStream(), charset));
        try {
            DataAdapter da = new StreamDataAdapter(in);
            da.computeStats();
            if (sampleStats.getN() == 0) {
                throw new ZeroException(LocalizedFormats.URL_CONTAINS_NO_DATA, url);
            }
            // new adapter for the second pass
            in = new BufferedReader(new InputStreamReader(url.openStream(), charset));
            fillBinStats(new StreamDataAdapter(in));
            loaded = true;
        } finally {
           try {
               in.close();
           } catch (IOException ex) { //NOPMD
               // ignore
           }
        }
    }

    
    public void load(File file) throws IOException, NullArgumentException {
        MathUtils.checkNotNull(file);
        Charset charset = Charset.forName(FILE_CHARSET);
        InputStream is = new FileInputStream(file);
        BufferedReader in = new BufferedReader(new InputStreamReader(is, charset));
        try {
            DataAdapter da = new StreamDataAdapter(in);
            da.computeStats();
            // new adapter for second pass
            is = new FileInputStream(file);
            in = new BufferedReader(new InputStreamReader(is, charset));
            fillBinStats(new StreamDataAdapter(in));
            loaded = true;
        } finally {
            try {
                in.close();
            } catch (IOException ex) { //NOPMD
                // ignore
            }
        }
    }

    
    private abstract class DataAdapter{

        
        public abstract void computeBinStats() throws IOException;

        
        public abstract void computeStats() throws IOException;

    }

    
    private class StreamDataAdapter extends DataAdapter{

        
        private BufferedReader inputStream;

        
        StreamDataAdapter(BufferedReader in){
            super();
            inputStream = in;
        }

        
        @Override
        public void computeBinStats() throws IOException {
            String str = null;
            double val = 0.0d;
            while ((str = inputStream.readLine()) != null) {
                val = Double.parseDouble(str);
                SummaryStatistics stats = binStats.get(findBin(val));
                stats.addValue(val);
            }

            inputStream.close();
            inputStream = null;
        }

        
        @Override
        public void computeStats() throws IOException {
            String str = null;
            double val = 0.0;
            sampleStats = new SummaryStatistics();
            while ((str = inputStream.readLine()) != null) {
                val = Double.parseDouble(str);
                sampleStats.addValue(val);
            }
            inputStream.close();
            inputStream = null;
        }
    }

    
    private class ArrayDataAdapter extends DataAdapter {

        
        private double[] inputArray;

        
        ArrayDataAdapter(double[] in) throws NullArgumentException {
            super();
            MathUtils.checkNotNull(in);
            inputArray = in;
        }

        
        @Override
        public void computeStats() throws IOException {
            sampleStats = new SummaryStatistics();
            for (int i = 0; i < inputArray.length; i++) {
                sampleStats.addValue(inputArray[i]);
            }
        }

        
        @Override
        public void computeBinStats() throws IOException {
            for (int i = 0; i < inputArray.length; i++) {
                SummaryStatistics stats =
                    binStats.get(findBin(inputArray[i]));
                stats.addValue(inputArray[i]);
            }
        }
    }

    
    private void fillBinStats(final DataAdapter da)
        throws IOException {
        // Set up grid
        min = sampleStats.getMin();
        max = sampleStats.getMax();
        delta = (max - min)/((double) binCount);

        // Initialize binStats ArrayList
        if (!binStats.isEmpty()) {
            binStats.clear();
        }
        for (int i = 0; i < binCount; i++) {
            SummaryStatistics stats = new SummaryStatistics();
            binStats.add(i,stats);
        }

        // Filling data in binStats Array
        da.computeBinStats();

        // Assign upperBounds based on bin counts
        upperBounds = new double[binCount];
        upperBounds[0] =
        ((double) binStats.get(0).getN()) / (double) sampleStats.getN();
        for (int i = 1; i < binCount-1; i++) {
            upperBounds[i] = upperBounds[i-1] +
            ((double) binStats.get(i).getN()) / (double) sampleStats.getN();
        }
        upperBounds[binCount-1] = 1.0d;
    }

    
    private int findBin(double value) {
        return FastMath.min(
                FastMath.max((int) FastMath.ceil((value - min) / delta) - 1, 0),
                binCount - 1);
    }

    
    public double getNextValue() throws MathIllegalStateException {

        if (!loaded) {
            throw new MathIllegalStateException(LocalizedFormats.DISTRIBUTION_NOT_LOADED);
        }

        return sample();
    }

    
    public StatisticalSummary getSampleStats() {
        return sampleStats;
    }

    
    public int getBinCount() {
        return binCount;
    }

    
    public List<SummaryStatistics> getBinStats() {
        return binStats;
    }

    
    public double[] getUpperBounds() {
        double[] binUpperBounds = new double[binCount];
        for (int i = 0; i < binCount - 1; i++) {
            binUpperBounds[i] = min + delta * (i + 1);
        }
        binUpperBounds[binCount - 1] = max;
        return binUpperBounds;
    }

    
    public double[] getGeneratorUpperBounds() {
        int len = upperBounds.length;
        double[] out = new double[len];
        System.arraycopy(upperBounds, 0, out, 0, len);
        return out;
    }

    
    public boolean isLoaded() {
        return loaded;
    }

    
    public void reSeed(long seed) {
        randomData.reSeed(seed);
    }

    // Distribution methods ---------------------------

    
    @Override
    public double probability(double x) {
        return 0;
    }

    
    public double density(double x) {
        if (x < min || x > max) {
            return 0d;
        }
        final int binIndex = findBin(x);
        final RealDistribution kernel = getKernel(binStats.get(binIndex));
        return kernel.density(x) * pB(binIndex) / kB(binIndex);
    }

    
    public double cumulativeProbability(double x) {
        if (x < min) {
            return 0d;
        } else if (x >= max) {
            return 1d;
        }
        final int binIndex = findBin(x);
        final double pBminus = pBminus(binIndex);
        final double pB = pB(binIndex);
        final RealDistribution kernel = k(x);
        if (kernel instanceof ConstantRealDistribution) {
            if (x < kernel.getNumericalMean()) {
                return pBminus;
            } else {
                return pBminus + pB;
            }
        }
        final double[] binBounds = getUpperBounds();
        final double kB = kB(binIndex);
        final double lower = binIndex == 0 ? min : binBounds[binIndex - 1];
        final double withinBinCum =
            (kernel.cumulativeProbability(x) -  kernel.cumulativeProbability(lower)) / kB;
        return pBminus + pB * withinBinCum;
    }

    
    @Override
    public double inverseCumulativeProbability(final double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }

        if (p == 0.0) {
            return getSupportLowerBound();
        }

        if (p == 1.0) {
            return getSupportUpperBound();
        }

        int i = 0;
        while (cumBinP(i) < p) {
            i++;
        }

        final RealDistribution kernel = getKernel(binStats.get(i));
        final double kB = kB(i);
        final double[] binBounds = getUpperBounds();
        final double lower = i == 0 ? min : binBounds[i - 1];
        final double kBminus = kernel.cumulativeProbability(lower);
        final double pB = pB(i);
        final double pBminus = pBminus(i);
        final double pCrit = p - pBminus;
        if (pCrit <= 0) {
            return lower;
        }
        return kernel.inverseCumulativeProbability(kBminus + pCrit * kB / pB);
    }

    
    public double getNumericalMean() {
       return sampleStats.getMean();
    }

    
    public double getNumericalVariance() {
        return sampleStats.getVariance();
    }

    
    public double getSupportLowerBound() {
       return min;
    }

    
    public double getSupportUpperBound() {
        return max;
    }

    
    public boolean isSupportLowerBoundInclusive() {
        return true;
    }

    
    public boolean isSupportUpperBoundInclusive() {
        return true;
    }

    
    public boolean isSupportConnected() {
        return true;
    }

    
    @Override
    public void reseedRandomGenerator(long seed) {
        randomData.reSeed(seed);
    }

    
    private double pB(int i) {
        return i == 0 ? upperBounds[0] :
            upperBounds[i] - upperBounds[i - 1];
    }

    
    private double pBminus(int i) {
        return i == 0 ? 0 : upperBounds[i - 1];
    }

    
    @SuppressWarnings("deprecation")
    private double kB(int i) {
        final double[] binBounds = getUpperBounds();
        final RealDistribution kernel = getKernel(binStats.get(i));
        return i == 0 ? kernel.cumulativeProbability(min, binBounds[0]) :
            kernel.cumulativeProbability(binBounds[i - 1], binBounds[i]);
    }

    
    private RealDistribution k(double x) {
        final int binIndex = findBin(x);
        return getKernel(binStats.get(binIndex));
    }

    
    private double cumBinP(int binIndex) {
        return upperBounds[binIndex];
    }

    
    protected RealDistribution getKernel(SummaryStatistics bStats) {
        if (bStats.getN() == 1 || bStats.getVariance() == 0) {
            return new ConstantRealDistribution(bStats.getMean());
        } else {
            return new NormalDistribution(randomData.getRandomGenerator(),
                bStats.getMean(), bStats.getStandardDeviation(),
                NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
        }
    }
}
