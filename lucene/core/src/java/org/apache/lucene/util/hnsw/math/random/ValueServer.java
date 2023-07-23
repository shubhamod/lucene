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
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.ZeroException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class ValueServer {

    
    public static final int DIGEST_MODE = 0;

    
    public static final int REPLAY_MODE = 1;

    
    public static final int UNIFORM_MODE = 2;

    
    public static final int EXPONENTIAL_MODE = 3;

    
    public static final int GAUSSIAN_MODE = 4;

    
    public static final int CONSTANT_MODE = 5;

    
    private int mode = 5;

    
    private URL valuesFileURL = null;

    
    private double mu = 0.0;

    
    private double sigma = 0.0;

    
    private EmpiricalDistribution empiricalDistribution = null;

    
    private BufferedReader filePointer = null;

    
    private final RandomDataGenerator randomData;

    // Data generation modes ======================================

    
    public ValueServer() {
        randomData = new RandomDataGenerator();
    }

    
    @Deprecated
    public ValueServer(RandomDataImpl randomData) {
        this.randomData = randomData.getDelegate();
    }

    
    public ValueServer(RandomGenerator generator) {
        this.randomData = new RandomDataGenerator(generator);
    }

    
    public double getNext() throws IOException, MathIllegalStateException, MathIllegalArgumentException {
        switch (mode) {
            case DIGEST_MODE: return getNextDigest();
            case REPLAY_MODE: return getNextReplay();
            case UNIFORM_MODE: return getNextUniform();
            case EXPONENTIAL_MODE: return getNextExponential();
            case GAUSSIAN_MODE: return getNextGaussian();
            case CONSTANT_MODE: return mu;
            default: throw new MathIllegalStateException(
                    LocalizedFormats.UNKNOWN_MODE,
                    mode,
                    "DIGEST_MODE",   DIGEST_MODE,   "REPLAY_MODE",      REPLAY_MODE,
                    "UNIFORM_MODE",  UNIFORM_MODE,  "EXPONENTIAL_MODE", EXPONENTIAL_MODE,
                    "GAUSSIAN_MODE", GAUSSIAN_MODE, "CONSTANT_MODE",    CONSTANT_MODE);
        }
    }

    
    public void fill(double[] values)
        throws IOException, MathIllegalStateException, MathIllegalArgumentException {
        for (int i = 0; i < values.length; i++) {
            values[i] = getNext();
        }
    }

    
    public double[] fill(int length)
        throws IOException, MathIllegalStateException, MathIllegalArgumentException {
        double[] out = new double[length];
        for (int i = 0; i < length; i++) {
            out[i] = getNext();
        }
        return out;
    }

    
    public void computeDistribution() throws IOException, ZeroException, NullArgumentException {
        computeDistribution(EmpiricalDistribution.DEFAULT_BIN_COUNT);
    }

    
    public void computeDistribution(int binCount) throws NullArgumentException, IOException, ZeroException {
        empiricalDistribution = new EmpiricalDistribution(binCount, randomData.getRandomGenerator());
        empiricalDistribution.load(valuesFileURL);
        mu = empiricalDistribution.getSampleStats().getMean();
        sigma = empiricalDistribution.getSampleStats().getStandardDeviation();
    }

    
    public int getMode() {
        return mode;
    }

    
    public void setMode(int mode) {
        this.mode = mode;
    }

    
    public URL getValuesFileURL() {
        return valuesFileURL;
    }

    
    public void setValuesFileURL(String url) throws MalformedURLException {
        this.valuesFileURL = new URL(url);
    }

    
    public void setValuesFileURL(URL url) {
        this.valuesFileURL = url;
    }

    
    public EmpiricalDistribution getEmpiricalDistribution() {
        return empiricalDistribution;
    }

    
    public void resetReplayFile() throws IOException {
        if (filePointer != null) {
            try {
                filePointer.close();
                filePointer = null;
            } catch (IOException ex) { //NOPMD
                // ignore
            }
        }
        filePointer = new BufferedReader(new InputStreamReader(valuesFileURL.openStream(), "UTF-8"));
    }

    
    public void closeReplayFile() throws IOException {
        if (filePointer != null) {
            filePointer.close();
            filePointer = null;
        }
    }

    
    public double getMu() {
        return mu;
    }

    
    public void setMu(double mu) {
        this.mu = mu;
    }

    
    public double getSigma() {
        return sigma;
    }

    
    public void setSigma(double sigma) {
        this.sigma = sigma;
    }

    
    public void reSeed(long seed) {
        randomData.reSeed(seed);
    }

    //------------- private methods ---------------------------------

    
    private double getNextDigest() throws MathIllegalStateException {
        if ((empiricalDistribution == null) ||
            (empiricalDistribution.getBinStats().size() == 0)) {
            throw new MathIllegalStateException(LocalizedFormats.DIGEST_NOT_INITIALIZED);
        }
        return empiricalDistribution.getNextValue();
    }

    
    private double getNextReplay() throws IOException, MathIllegalStateException {
        String str = null;
        if (filePointer == null) {
            resetReplayFile();
        }
        if ((str = filePointer.readLine()) == null) {
            // we have probably reached end of file, wrap around from EOF to BOF
            closeReplayFile();
            resetReplayFile();
            if ((str = filePointer.readLine()) == null) {
                throw new MathIllegalStateException(LocalizedFormats.URL_CONTAINS_NO_DATA,
                                                    valuesFileURL);
            }
        }
        return Double.parseDouble(str);
    }

    
    private double getNextUniform() throws MathIllegalArgumentException {
        return randomData.nextUniform(0, 2 * mu);
    }

    
    private double getNextExponential() throws MathIllegalArgumentException {
        return randomData.nextExponential(mu);
    }

    
    private double getNextGaussian() throws MathIllegalArgumentException {
        return randomData.nextGaussian(mu, sigma);
    }

}
