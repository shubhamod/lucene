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

import java.io.Serializable;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.SecureRandom;
import java.util.Collection;

import org.apache.lucene.util.hnsw.math.distribution.BetaDistribution;
import org.apache.lucene.util.hnsw.math.distribution.BinomialDistribution;
import org.apache.lucene.util.hnsw.math.distribution.CauchyDistribution;
import org.apache.lucene.util.hnsw.math.distribution.ChiSquaredDistribution;
import org.apache.lucene.util.hnsw.math.distribution.ExponentialDistribution;
import org.apache.lucene.util.hnsw.math.distribution.FDistribution;
import org.apache.lucene.util.hnsw.math.distribution.GammaDistribution;
import org.apache.lucene.util.hnsw.math.distribution.HypergeometricDistribution;
import org.apache.lucene.util.hnsw.math.distribution.PascalDistribution;
import org.apache.lucene.util.hnsw.math.distribution.PoissonDistribution;
import org.apache.lucene.util.hnsw.math.distribution.TDistribution;
import org.apache.lucene.util.hnsw.math.distribution.WeibullDistribution;
import org.apache.lucene.util.hnsw.math.distribution.ZipfDistribution;
import org.apache.lucene.util.hnsw.math.distribution.UniformIntegerDistribution;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.NotANumberException;
import org.apache.lucene.util.hnsw.math.exception.NotFiniteNumberException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathArrays;


public class RandomDataGenerator implements RandomData, Serializable {

    
    private static final long serialVersionUID = -626730818244969716L;

    
    private RandomGenerator rand = null;

    
    private RandomGenerator secRand = null;

    
    public RandomDataGenerator() {
    }

    
    public RandomDataGenerator(RandomGenerator rand) {
        this.rand = rand;
    }

    
    public String nextHexString(int len) throws NotStrictlyPositiveException {
        if (len <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.LENGTH, len);
        }

        // Get a random number generator
        RandomGenerator ran = getRandomGenerator();

        // Initialize output buffer
        StringBuilder outBuffer = new StringBuilder();

        // Get int(len/2)+1 random bytes
        byte[] randomBytes = new byte[(len / 2) + 1];
        ran.nextBytes(randomBytes);

        // Convert each byte to 2 hex digits
        for (int i = 0; i < randomBytes.length; i++) {
            Integer c = Integer.valueOf(randomBytes[i]);

            /*
             * Add 128 to byte value to make interval 0-255 before doing hex
             * conversion. This guarantees <= 2 hex digits from toHexString()
             * toHexString would otherwise add 2^32 to negative arguments.
             */
            String hex = Integer.toHexString(c.intValue() + 128);

            // Make sure we add 2 hex digits for each byte
            if (hex.length() == 1) {
                hex = "0" + hex;
            }
            outBuffer.append(hex);
        }
        return outBuffer.toString().substring(0, len);
    }

    
    public int nextInt(final int lower, final int upper) throws NumberIsTooLargeException {
        return new UniformIntegerDistribution(getRandomGenerator(), lower, upper).sample();
    }

    
    public long nextLong(final long lower, final long upper) throws NumberIsTooLargeException {
        if (lower >= upper) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND,
                                                lower, upper, false);
        }
        final long max = (upper - lower) + 1;
        if (max <= 0) {
            // the range is too wide to fit in a positive long (larger than 2^63); as it covers
            // more than half the long range, we use directly a simple rejection method
            final RandomGenerator rng = getRandomGenerator();
            while (true) {
                final long r = rng.nextLong();
                if (r >= lower && r <= upper) {
                    return r;
                }
            }
        } else if (max < Integer.MAX_VALUE){
            // we can shift the range and generate directly a positive int
            return lower + getRandomGenerator().nextInt((int) max);
        } else {
            // we can shift the range and generate directly a positive long
            return lower + nextLong(getRandomGenerator(), max);
        }
    }

    
    private static long nextLong(final RandomGenerator rng, final long n) throws IllegalArgumentException {
        if (n > 0) {
            final byte[] byteArray = new byte[8];
            long bits;
            long val;
            do {
                rng.nextBytes(byteArray);
                bits = 0;
                for (final byte b : byteArray) {
                    bits = (bits << 8) | (((long) b) & 0xffL);
                }
                bits &= 0x7fffffffffffffffL;
                val  = bits % n;
            } while (bits - val + (n - 1) < 0);
            return val;
        }
        throw new NotStrictlyPositiveException(n);
    }

    
    public String nextSecureHexString(int len) throws NotStrictlyPositiveException {
        if (len <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.LENGTH, len);
        }

        // Get SecureRandom and setup Digest provider
        final RandomGenerator secRan = getSecRan();
        MessageDigest alg = null;
        try {
            alg = MessageDigest.getInstance("SHA-1");
        } catch (NoSuchAlgorithmException ex) {
            // this should never happen
            throw new MathInternalError(ex);
        }
        alg.reset();

        // Compute number of iterations required (40 bytes each)
        int numIter = (len / 40) + 1;

        StringBuilder outBuffer = new StringBuilder();
        for (int iter = 1; iter < numIter + 1; iter++) {
            byte[] randomBytes = new byte[40];
            secRan.nextBytes(randomBytes);
            alg.update(randomBytes);

            // Compute hash -- will create 20-byte binary hash
            byte[] hash = alg.digest();

            // Loop over the hash, converting each byte to 2 hex digits
            for (int i = 0; i < hash.length; i++) {
                Integer c = Integer.valueOf(hash[i]);

                /*
                 * Add 128 to byte value to make interval 0-255 This guarantees
                 * <= 2 hex digits from toHexString() toHexString would
                 * otherwise add 2^32 to negative arguments
                 */
                String hex = Integer.toHexString(c.intValue() + 128);

                // Keep strings uniform length -- guarantees 40 bytes
                if (hex.length() == 1) {
                    hex = "0" + hex;
                }
                outBuffer.append(hex);
            }
        }
        return outBuffer.toString().substring(0, len);
    }

    
    public int nextSecureInt(final int lower, final int upper) throws NumberIsTooLargeException {
        return new UniformIntegerDistribution(getSecRan(), lower, upper).sample();
    }

    
    public long nextSecureLong(final long lower, final long upper) throws NumberIsTooLargeException {
        if (lower >= upper) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND,
                                                lower, upper, false);
        }
        final RandomGenerator rng = getSecRan();
        final long max = (upper - lower) + 1;
        if (max <= 0) {
            // the range is too wide to fit in a positive long (larger than 2^63); as it covers
            // more than half the long range, we use directly a simple rejection method
            while (true) {
                final long r = rng.nextLong();
                if (r >= lower && r <= upper) {
                    return r;
                }
            }
        } else if (max < Integer.MAX_VALUE){
            // we can shift the range and generate directly a positive int
            return lower + rng.nextInt((int) max);
        } else {
            // we can shift the range and generate directly a positive long
            return lower + nextLong(rng, max);
        }
    }

    
    public long nextPoisson(double mean) throws NotStrictlyPositiveException {
        return new PoissonDistribution(getRandomGenerator(), mean,
                PoissonDistribution.DEFAULT_EPSILON,
                PoissonDistribution.DEFAULT_MAX_ITERATIONS).sample();
    }

    
    public double nextGaussian(double mu, double sigma) throws NotStrictlyPositiveException {
        if (sigma <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.STANDARD_DEVIATION, sigma);
        }
        return sigma * getRandomGenerator().nextGaussian() + mu;
    }

    
    public double nextExponential(double mean) throws NotStrictlyPositiveException {
        return new ExponentialDistribution(getRandomGenerator(), mean,
                ExponentialDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY).sample();
    }

    
    public double nextGamma(double shape, double scale) throws NotStrictlyPositiveException {
        return new GammaDistribution(getRandomGenerator(),shape, scale,
                GammaDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY).sample();
    }

    
    public int nextHypergeometric(int populationSize, int numberOfSuccesses, int sampleSize) throws NotPositiveException, NotStrictlyPositiveException, NumberIsTooLargeException {
        return new HypergeometricDistribution(getRandomGenerator(),populationSize,
                numberOfSuccesses, sampleSize).sample();
    }

    
    public int nextPascal(int r, double p) throws NotStrictlyPositiveException, OutOfRangeException {
        return new PascalDistribution(getRandomGenerator(), r, p).sample();
    }

    
    public double nextT(double df) throws NotStrictlyPositiveException {
        return new TDistribution(getRandomGenerator(), df,
                TDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY).sample();
    }

    
    public double nextWeibull(double shape, double scale) throws NotStrictlyPositiveException {
        return new WeibullDistribution(getRandomGenerator(), shape, scale,
                WeibullDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY).sample();
    }

    
    public int nextZipf(int numberOfElements, double exponent) throws NotStrictlyPositiveException {
        return new ZipfDistribution(getRandomGenerator(), numberOfElements, exponent).sample();
    }

    
    public double nextBeta(double alpha, double beta) {
        return new BetaDistribution(getRandomGenerator(), alpha, beta,
                BetaDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY).sample();
    }

    
    public int nextBinomial(int numberOfTrials, double probabilityOfSuccess) {
        return new BinomialDistribution(getRandomGenerator(), numberOfTrials, probabilityOfSuccess).sample();
    }

    
    public double nextCauchy(double median, double scale) {
        return new CauchyDistribution(getRandomGenerator(), median, scale,
                CauchyDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY).sample();
    }

    
    public double nextChiSquare(double df) {
        return new ChiSquaredDistribution(getRandomGenerator(), df,
                ChiSquaredDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY).sample();
    }

    
    public double nextF(double numeratorDf, double denominatorDf) throws NotStrictlyPositiveException {
        return new FDistribution(getRandomGenerator(), numeratorDf, denominatorDf,
                FDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY).sample();
    }

    
    public double nextUniform(double lower, double upper)
        throws NumberIsTooLargeException, NotFiniteNumberException, NotANumberException {
        return nextUniform(lower, upper, false);
    }

    
    public double nextUniform(double lower, double upper, boolean lowerInclusive)
        throws NumberIsTooLargeException, NotFiniteNumberException, NotANumberException {

        if (lower >= upper) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND,
                                                lower, upper, false);
        }

        if (Double.isInfinite(lower)) {
            throw new NotFiniteNumberException(LocalizedFormats.INFINITE_BOUND, lower);
        }
        if (Double.isInfinite(upper)) {
            throw new NotFiniteNumberException(LocalizedFormats.INFINITE_BOUND, upper);
        }

        if (Double.isNaN(lower) || Double.isNaN(upper)) {
            throw new NotANumberException();
        }

        final RandomGenerator generator = getRandomGenerator();

        // ensure nextDouble() isn't 0.0
        double u = generator.nextDouble();
        while (!lowerInclusive && u <= 0.0) {
            u = generator.nextDouble();
        }

        return u * upper + (1.0 - u) * lower;
    }

    
    public int[] nextPermutation(int n, int k)
        throws NumberIsTooLargeException, NotStrictlyPositiveException {
        if (k > n) {
            throw new NumberIsTooLargeException(LocalizedFormats.PERMUTATION_EXCEEDS_N,
                                                k, n, true);
        }
        if (k <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.PERMUTATION_SIZE,
                                                   k);
        }

        int[] index = MathArrays.natural(n);
        MathArrays.shuffle(index, getRandomGenerator());

        // Return a new array containing the first "k" entries of "index".
        return MathArrays.copyOf(index, k);
    }

    
    public Object[] nextSample(Collection<?> c, int k) throws NumberIsTooLargeException, NotStrictlyPositiveException {

        int len = c.size();
        if (k > len) {
            throw new NumberIsTooLargeException(LocalizedFormats.SAMPLE_SIZE_EXCEEDS_COLLECTION_SIZE,
                                                k, len, true);
        }
        if (k <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_SAMPLES, k);
        }

        Object[] objects = c.toArray();
        int[] index = nextPermutation(len, k);
        Object[] result = new Object[k];
        for (int i = 0; i < k; i++) {
            result[i] = objects[index[i]];
        }
        return result;
    }



    
    public void reSeed(long seed) {
       getRandomGenerator().setSeed(seed);
    }

    
    public void reSeedSecure() {
        getSecRan().setSeed(System.currentTimeMillis());
    }

    
    public void reSeedSecure(long seed) {
        getSecRan().setSeed(seed);
    }

    
    public void reSeed() {
        getRandomGenerator().setSeed(System.currentTimeMillis() + System.identityHashCode(this));
    }

    
    public void setSecureAlgorithm(String algorithm, String provider)
            throws NoSuchAlgorithmException, NoSuchProviderException {
        secRand = RandomGeneratorFactory.createRandomGenerator(SecureRandom.getInstance(algorithm, provider));
    }

    
    public RandomGenerator getRandomGenerator() {
        if (rand == null) {
            initRan();
        }
        return rand;
    }

    
    private void initRan() {
        rand = new Well19937c(System.currentTimeMillis() + System.identityHashCode(this));
    }

    
    private RandomGenerator getSecRan() {
        if (secRand == null) {
            secRand = RandomGeneratorFactory.createRandomGenerator(new SecureRandom());
            secRand.setSeed(System.currentTimeMillis() + System.identityHashCode(this));
        }
        return secRand;
    }
}
