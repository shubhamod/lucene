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
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.util.Collection;

import org.apache.lucene.util.hnsw.math.distribution.IntegerDistribution;
import org.apache.lucene.util.hnsw.math.distribution.RealDistribution;
import org.apache.lucene.util.hnsw.math.exception.NotANumberException;
import org.apache.lucene.util.hnsw.math.exception.NotFiniteNumberException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;


@Deprecated
public class RandomDataImpl implements RandomData, Serializable {

    
    private static final long serialVersionUID = -626730818244969716L;

    
    private final RandomDataGenerator delegate;

    
    public RandomDataImpl() {
        delegate = new RandomDataGenerator();
    }

    
    public RandomDataImpl(RandomGenerator rand) {
        delegate = new RandomDataGenerator(rand);
    }

    
    @Deprecated
    RandomDataGenerator getDelegate() {
        return delegate;
    }

    
    public String nextHexString(int len) throws NotStrictlyPositiveException {
        return delegate.nextHexString(len);
    }

    
    public int nextInt(int lower, int upper) throws NumberIsTooLargeException {
       return delegate.nextInt(lower, upper);
    }

    
    public long nextLong(long lower, long upper) throws NumberIsTooLargeException {
        return delegate.nextLong(lower, upper);
    }

    
    public String nextSecureHexString(int len) throws NotStrictlyPositiveException {
        return delegate.nextSecureHexString(len);
    }

    
    public int nextSecureInt(int lower, int upper) throws NumberIsTooLargeException {
        return delegate.nextSecureInt(lower, upper);
    }

    
    public long nextSecureLong(long lower, long upper) throws NumberIsTooLargeException {
        return delegate.nextSecureLong(lower,upper);
    }

    
    public long nextPoisson(double mean) throws NotStrictlyPositiveException {
        return delegate.nextPoisson(mean);
    }

    
    public double nextGaussian(double mu, double sigma) throws NotStrictlyPositiveException {
        return delegate.nextGaussian(mu,sigma);
    }

    
    public double nextExponential(double mean) throws NotStrictlyPositiveException {
        return delegate.nextExponential(mean);
    }

    
    public double nextUniform(double lower, double upper)
        throws NumberIsTooLargeException, NotFiniteNumberException, NotANumberException {
        return delegate.nextUniform(lower, upper);
    }

    
    public double nextUniform(double lower, double upper, boolean lowerInclusive)
        throws NumberIsTooLargeException, NotFiniteNumberException, NotANumberException {
        return delegate.nextUniform(lower, upper, lowerInclusive);
    }

    
    public double nextBeta(double alpha, double beta) {
        return delegate.nextBeta(alpha, beta);
    }

    
    public int nextBinomial(int numberOfTrials, double probabilityOfSuccess) {
        return delegate.nextBinomial(numberOfTrials, probabilityOfSuccess);
    }

    
    public double nextCauchy(double median, double scale) {
        return delegate.nextCauchy(median, scale);
    }

    
    public double nextChiSquare(double df) {
       return delegate.nextChiSquare(df);
    }

    
    public double nextF(double numeratorDf, double denominatorDf) throws NotStrictlyPositiveException {
        return delegate.nextF(numeratorDf, denominatorDf);
    }

    
    public double nextGamma(double shape, double scale) throws NotStrictlyPositiveException {
        return delegate.nextGamma(shape, scale);
    }

    
    public int nextHypergeometric(int populationSize, int numberOfSuccesses, int sampleSize)
        throws NotPositiveException, NotStrictlyPositiveException, NumberIsTooLargeException {
        return delegate.nextHypergeometric(populationSize, numberOfSuccesses, sampleSize);
    }

    
    public int nextPascal(int r, double p)
        throws NotStrictlyPositiveException, OutOfRangeException {
        return delegate.nextPascal(r, p);
    }

    
    public double nextT(double df) throws NotStrictlyPositiveException {
        return delegate.nextT(df);
    }

    
    public double nextWeibull(double shape, double scale) throws NotStrictlyPositiveException {
        return delegate.nextWeibull(shape, scale);
    }

    
    public int nextZipf(int numberOfElements, double exponent) throws NotStrictlyPositiveException {
        return delegate.nextZipf(numberOfElements, exponent);
    }


    
    public void reSeed(long seed) {
        delegate.reSeed(seed);
    }

    
    public void reSeedSecure() {
        delegate.reSeedSecure();
    }

    
    public void reSeedSecure(long seed) {
        delegate.reSeedSecure(seed);
    }

    
    public void reSeed() {
        delegate.reSeed();
    }

    
    public void setSecureAlgorithm(String algorithm, String provider)
            throws NoSuchAlgorithmException, NoSuchProviderException {
       delegate.setSecureAlgorithm(algorithm, provider);
    }

    
    public int[] nextPermutation(int n, int k)
        throws NotStrictlyPositiveException, NumberIsTooLargeException {
        return delegate.nextPermutation(n, k);
    }

    
    public Object[] nextSample(Collection<?> c, int k)
        throws NotStrictlyPositiveException, NumberIsTooLargeException {
        return delegate.nextSample(c, k);
    }

    
    @Deprecated
    public double nextInversionDeviate(RealDistribution distribution)
        throws MathIllegalArgumentException {
        return distribution.inverseCumulativeProbability(nextUniform(0, 1));

    }

    
    @Deprecated
    public int nextInversionDeviate(IntegerDistribution distribution)
        throws MathIllegalArgumentException {
        return distribution.inverseCumulativeProbability(nextUniform(0, 1));
    }

}
