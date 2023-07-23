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
import java.util.Collection;

import org.apache.lucene.util.hnsw.math.exception.NotANumberException;
import org.apache.lucene.util.hnsw.math.exception.NotFiniteNumberException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;


@Deprecated
public interface RandomData {
    
    String nextHexString(int len) throws NotStrictlyPositiveException;

    
    int nextInt(int lower, int upper) throws NumberIsTooLargeException;

    
    long nextLong(long lower, long upper) throws NumberIsTooLargeException;

    
    String nextSecureHexString(int len) throws NotStrictlyPositiveException;

    
    int nextSecureInt(int lower, int upper) throws NumberIsTooLargeException;

    
    long nextSecureLong(long lower, long upper) throws NumberIsTooLargeException;

    
    long nextPoisson(double mean) throws NotStrictlyPositiveException;

    
    double nextGaussian(double mu, double sigma) throws NotStrictlyPositiveException;

    
    double nextExponential(double mean) throws NotStrictlyPositiveException;

    
    double nextUniform(double lower, double upper)
        throws NumberIsTooLargeException, NotFiniteNumberException, NotANumberException;

    
    double nextUniform(double lower, double upper, boolean lowerInclusive)
        throws NumberIsTooLargeException, NotFiniteNumberException, NotANumberException;

    
    int[] nextPermutation(int n, int k)
        throws NumberIsTooLargeException, NotStrictlyPositiveException;

    
    Object[] nextSample(Collection<?> c, int k)
        throws NumberIsTooLargeException, NotStrictlyPositiveException;

}
