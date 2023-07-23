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
package org.apache.lucene.util.hnsw.math;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;


public interface RealFieldElement<T> extends FieldElement<T> {

    
    double getReal();

    
    T add(double a);

    
    T subtract(double a);

    
    T multiply(double a);

    
    T divide(double a);

    
    T remainder(double a);

    
    T remainder(T a)
        throws DimensionMismatchException;

    
    T abs();

    
    T ceil();

    
    T floor();

    
    T rint();

    
    long round();

    
    T signum();

    
    T copySign(T sign);

    
    T copySign(double sign);

    
    T scalb(int n);

    
    T hypot(T y)
        throws DimensionMismatchException;

    
    T reciprocal();

    
    T sqrt();

    
    T cbrt();

    
    T rootN(int n);

    
    T pow(double p);

    
    T pow(int n);

    
    T pow(T e)
        throws DimensionMismatchException;

    
    T exp();

    
    T expm1();

    
    T log();

    
    T log1p();

//    TODO: add this method in 4.0, as it is not possible to do it in 3.2
//          due to incompatibility of the return type in the Dfp class
//    
//    T log10();

    
    T cos();

    
    T sin();

    
    T tan();

    
    T acos();

    
    T asin();

    
    T atan();

    
    T atan2(T x)
        throws DimensionMismatchException;

    
    T cosh();

    
    T sinh();

    
    T tanh();

    
    T acosh();

    
    T asinh();

    
    T atanh();

    
    T linearCombination(T[] a, T[] b)
        throws DimensionMismatchException;

    
    T linearCombination(double[] a, T[] b)
        throws DimensionMismatchException;

    
    T linearCombination(T a1, T b1, T a2, T b2);

    
    T linearCombination(double a1, T b1, double a2, T b2);

    
    T linearCombination(T a1, T b1, T a2, T b2, T a3, T b3);

    
    T linearCombination(double a1, T b1,  double a2, T b2, double a3, T b3);

    
    T linearCombination(T a1, T b1, T a2, T b2, T a3, T b3, T a4, T b4);

    
    T linearCombination(double a1, T b1, double a2, T b2, double a3, T b3, double a4, T b4);

}
