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
package org.apache.lucene.util.hnsw.math.linear;

import org.apache.lucene.util.hnsw.math.Field;
import org.apache.lucene.util.hnsw.math.FieldElement;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathArithmeticException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;


public interface FieldVector<T extends FieldElement<T>>  {

    
    Field<T> getField();

    
    FieldVector<T> copy();

    
    FieldVector<T> add(FieldVector<T> v) throws DimensionMismatchException;

    
    FieldVector<T> subtract(FieldVector<T> v) throws DimensionMismatchException;

    
    FieldVector<T> mapAdd(T d) throws NullArgumentException;

    
    FieldVector<T> mapAddToSelf(T d) throws NullArgumentException;

    
    FieldVector<T> mapSubtract(T d) throws NullArgumentException;

    
    FieldVector<T> mapSubtractToSelf(T d) throws NullArgumentException;

    
    FieldVector<T> mapMultiply(T d) throws NullArgumentException;

    
    FieldVector<T> mapMultiplyToSelf(T d) throws NullArgumentException;

    
    FieldVector<T> mapDivide(T d)
        throws NullArgumentException, MathArithmeticException;

    
    FieldVector<T> mapDivideToSelf(T d)
        throws NullArgumentException, MathArithmeticException;

    
    FieldVector<T> mapInv() throws MathArithmeticException;

    
    FieldVector<T> mapInvToSelf() throws MathArithmeticException;

    
    FieldVector<T> ebeMultiply(FieldVector<T> v)
        throws DimensionMismatchException;

    
    FieldVector<T> ebeDivide(FieldVector<T> v)
        throws DimensionMismatchException, MathArithmeticException;

    
    @Deprecated
    T[] getData();

    
    T dotProduct(FieldVector<T> v) throws DimensionMismatchException;

    
    FieldVector<T> projection(FieldVector<T> v)
        throws DimensionMismatchException, MathArithmeticException;

    
    FieldMatrix<T> outerProduct(FieldVector<T> v);

    
    T getEntry(int index) throws OutOfRangeException;

    
    void setEntry(int index, T value) throws OutOfRangeException;

    
    int getDimension();

    
    FieldVector<T> append(FieldVector<T> v);

    
    FieldVector<T> append(T d);

    
    FieldVector<T> getSubVector(int index, int n)
        throws OutOfRangeException, NotPositiveException;

    
    void setSubVector(int index, FieldVector<T> v) throws OutOfRangeException;

    
    void set(T value);

    
    T[] toArray();

}
