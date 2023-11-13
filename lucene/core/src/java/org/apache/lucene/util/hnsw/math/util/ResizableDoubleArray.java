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
package org.apache.lucene.util.hnsw.math.util;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalStateException;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class ResizableDoubleArray implements DoubleArray, Serializable {
    
    @Deprecated
    public static final int ADDITIVE_MODE = 1;
    
    @Deprecated
    public static final int MULTIPLICATIVE_MODE = 0;
    
    private static final long serialVersionUID = -3485529955529426875L;

    
    private static final int DEFAULT_INITIAL_CAPACITY = 16;
    
    private static final double DEFAULT_EXPANSION_FACTOR = 2.0;
    
    private static final double DEFAULT_CONTRACTION_DELTA = 0.5;

    
    private double contractionCriterion = 2.5;

    
    private double expansionFactor = 2.0;

    
    private ExpansionMode expansionMode = ExpansionMode.MULTIPLICATIVE;

    
    private double[] internalArray;

    
    private int numElements = 0;

    
    private int startIndex = 0;

    
    public enum ExpansionMode {
        
        MULTIPLICATIVE,
        
        ADDITIVE
    }

    
    public ResizableDoubleArray() {
        this(DEFAULT_INITIAL_CAPACITY);
    }

    
    public ResizableDoubleArray(int initialCapacity)
        throws MathIllegalArgumentException {
        this(initialCapacity, DEFAULT_EXPANSION_FACTOR);
    }

    
    public ResizableDoubleArray(double[] initialArray) {
        this(DEFAULT_INITIAL_CAPACITY,
             DEFAULT_EXPANSION_FACTOR,
             DEFAULT_CONTRACTION_DELTA + DEFAULT_EXPANSION_FACTOR,
             ExpansionMode.MULTIPLICATIVE,
             initialArray);
    }

    
    @Deprecated
    public ResizableDoubleArray(int initialCapacity,
                                float expansionFactor)
        throws MathIllegalArgumentException {
        this(initialCapacity,
             (double) expansionFactor);
    }

    
    public ResizableDoubleArray(int initialCapacity,
                                double expansionFactor)
        throws MathIllegalArgumentException {
        this(initialCapacity,
             expansionFactor,
             DEFAULT_CONTRACTION_DELTA + expansionFactor);
    }

    
    @Deprecated
    public ResizableDoubleArray(int initialCapacity,
                                float expansionFactor,
                                float contractionCriteria)
        throws MathIllegalArgumentException {
        this(initialCapacity,
             (double) expansionFactor,
             (double) contractionCriteria);
    }

    
    public ResizableDoubleArray(int initialCapacity,
                                double expansionFactor,
                                double contractionCriterion)
        throws MathIllegalArgumentException {
        this(initialCapacity,
             expansionFactor,
             contractionCriterion,
             ExpansionMode.MULTIPLICATIVE,
             null);
    }

    
    @Deprecated
    public ResizableDoubleArray(int initialCapacity, float expansionFactor,
            float contractionCriteria, int expansionMode) throws MathIllegalArgumentException {
        this(initialCapacity,
             expansionFactor,
             contractionCriteria,
             expansionMode == ADDITIVE_MODE ?
             ExpansionMode.ADDITIVE :
             ExpansionMode.MULTIPLICATIVE,
             null);
        // XXX Just ot retain the expected failure in a unit test.
        // With the new "enum", that test will become obsolete.
        setExpansionMode(expansionMode);
    }

    
    public ResizableDoubleArray(int initialCapacity,
                                double expansionFactor,
                                double contractionCriterion,
                                ExpansionMode expansionMode,
                                double ... data)
        throws MathIllegalArgumentException {
        if (initialCapacity <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.INITIAL_CAPACITY_NOT_POSITIVE,
                                                   initialCapacity);
        }
        checkContractExpand(contractionCriterion, expansionFactor);

        this.expansionFactor = expansionFactor;
        this.contractionCriterion = contractionCriterion;
        this.expansionMode = expansionMode;
        internalArray = new double[initialCapacity];
        numElements = 0;
        startIndex = 0;

        if (data != null && data.length > 0) {
            addElements(data);
        }
    }

    
    public ResizableDoubleArray(ResizableDoubleArray original)
        throws NullArgumentException {
        MathUtils.checkNotNull(original);
        copy(original, this);
    }

    
    public synchronized void addElement(double value) {
        if (internalArray.length <= startIndex + numElements) {
            expand();
        }
        internalArray[startIndex + numElements++] = value;
    }

    
    public synchronized void addElements(double[] values) {
        final double[] tempArray = new double[numElements + values.length + 1];
        System.arraycopy(internalArray, startIndex, tempArray, 0, numElements);
        System.arraycopy(values, 0, tempArray, numElements, values.length);
        internalArray = tempArray;
        startIndex = 0;
        numElements += values.length;
    }

    
    public synchronized double addElementRolling(double value) {
        double discarded = internalArray[startIndex];

        if ((startIndex + (numElements + 1)) > internalArray.length) {
            expand();
        }
        // Increment the start index
        startIndex += 1;

        // Add the new value
        internalArray[startIndex + (numElements - 1)] = value;

        // Check the contraction criterion.
        if (shouldContract()) {
            contract();
        }
        return discarded;
    }

    
    public synchronized double substituteMostRecentElement(double value)
        throws MathIllegalStateException {
        if (numElements < 1) {
            throw new MathIllegalStateException(
                    LocalizedFormats.CANNOT_SUBSTITUTE_ELEMENT_FROM_EMPTY_ARRAY);
        }

        final int substIndex = startIndex + (numElements - 1);
        final double discarded = internalArray[substIndex];

        internalArray[substIndex] = value;

        return discarded;
    }

    
    @Deprecated
    protected void checkContractExpand(float contraction, float expansion)
        throws MathIllegalArgumentException {
        checkContractExpand((double) contraction,
                            (double) expansion);
    }

    
    protected void checkContractExpand(double contraction,
                                       double expansion)
        throws NumberIsTooSmallException {
        if (contraction < expansion) {
            final NumberIsTooSmallException e = new NumberIsTooSmallException(contraction, 1, true);
            e.getContext().addMessage(LocalizedFormats.CONTRACTION_CRITERIA_SMALLER_THAN_EXPANSION_FACTOR,
                                      contraction, expansion);
            throw e;
        }

        if (contraction <= 1) {
            final NumberIsTooSmallException e = new NumberIsTooSmallException(contraction, 1, false);
            e.getContext().addMessage(LocalizedFormats.CONTRACTION_CRITERIA_SMALLER_THAN_ONE,
                                      contraction);
            throw e;
        }

        if (expansion <= 1) {
            final NumberIsTooSmallException e = new NumberIsTooSmallException(contraction, 1, false);
            e.getContext().addMessage(LocalizedFormats.EXPANSION_FACTOR_SMALLER_THAN_ONE,
                                      expansion);
            throw e;
        }
    }

    
    public synchronized void clear() {
        numElements = 0;
        startIndex = 0;
    }

    
    public synchronized void contract() {
        final double[] tempArray = new double[numElements + 1];

        // Copy and swap - copy only the element array from the src array.
        System.arraycopy(internalArray, startIndex, tempArray, 0, numElements);
        internalArray = tempArray;

        // Reset the start index to zero
        startIndex = 0;
    }

    
    public synchronized void discardFrontElements(int i)
        throws MathIllegalArgumentException {
        discardExtremeElements(i,true);
    }

    
    public synchronized void discardMostRecentElements(int i)
        throws MathIllegalArgumentException {
        discardExtremeElements(i,false);
    }

    
    private synchronized void discardExtremeElements(int i,
                                                     boolean front)
        throws MathIllegalArgumentException {
        if (i > numElements) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.TOO_MANY_ELEMENTS_TO_DISCARD_FROM_ARRAY,
                    i, numElements);
       } else if (i < 0) {
           throw new MathIllegalArgumentException(
                   LocalizedFormats.CANNOT_DISCARD_NEGATIVE_NUMBER_OF_ELEMENTS,
                   i);
        } else {
            // "Subtract" this number of discarded from numElements
            numElements -= i;
            if (front) {
                startIndex += i;
            }
        }
        if (shouldContract()) {
            contract();
        }
    }

    
    protected synchronized void expand() {
        // notice the use of FastMath.ceil(), this guarantees that we will always
        // have an array of at least currentSize + 1.   Assume that the
        // current initial capacity is 1 and the expansion factor
        // is 1.000000000000000001.  The newly calculated size will be
        // rounded up to 2 after the multiplication is performed.
        int newSize = 0;
        if (expansionMode == ExpansionMode.MULTIPLICATIVE) {
            newSize = (int) FastMath.ceil(internalArray.length * expansionFactor);
        } else {
            newSize = (int) (internalArray.length + FastMath.round(expansionFactor));
        }
        final double[] tempArray = new double[newSize];

        // Copy and swap
        System.arraycopy(internalArray, 0, tempArray, 0, internalArray.length);
        internalArray = tempArray;
    }

    
    private synchronized void expandTo(int size) {
        final double[] tempArray = new double[size];
        // Copy and swap
        System.arraycopy(internalArray, 0, tempArray, 0, internalArray.length);
        internalArray = tempArray;
    }

    
    @Deprecated
    public float getContractionCriteria() {
        return (float) getContractionCriterion();
    }

    
    public double getContractionCriterion() {
        return contractionCriterion;
    }

    
    public synchronized double getElement(int index) {
        if (index >= numElements) {
            throw new ArrayIndexOutOfBoundsException(index);
        } else if (index >= 0) {
            return internalArray[startIndex + index];
        } else {
            throw new ArrayIndexOutOfBoundsException(index);
        }
    }

     
    public synchronized double[] getElements() {
        final double[] elementArray = new double[numElements];
        System.arraycopy(internalArray, startIndex, elementArray, 0, numElements);
        return elementArray;
    }

    
    @Deprecated
    public float getExpansionFactor() {
        return (float) expansionFactor;
    }

    
    @Deprecated
    public int getExpansionMode() {
        synchronized (this) {
            switch (expansionMode) {
                case MULTIPLICATIVE:
                    return MULTIPLICATIVE_MODE;
                case ADDITIVE:
                    return ADDITIVE_MODE;
                default:
                    throw new MathInternalError(); // Should never happen.
            }
        }
    }

    
    @Deprecated
    synchronized int getInternalLength() {
        return internalArray.length;
    }

    
    public int getCapacity() {
        return internalArray.length;
    }

    
    public synchronized int getNumElements() {
        return numElements;
    }

    
    @Deprecated
    public synchronized double[] getInternalValues() {
        return internalArray;
    }

    
    protected double[] getArrayRef() {
        return internalArray;
    }

    
    protected int getStartIndex() {
        return startIndex;
    }

    
    @Deprecated
    public void setContractionCriteria(float contractionCriteria)
        throws MathIllegalArgumentException {
        checkContractExpand(contractionCriteria, getExpansionFactor());
        synchronized(this) {
            this.contractionCriterion = contractionCriteria;
        }
    }

    
    public double compute(MathArrays.Function f) {
        final double[] array;
        final int start;
        final int num;
        synchronized(this) {
            array = internalArray;
            start = startIndex;
            num   = numElements;
        }
        return f.evaluate(array, start, num);
    }

    
    public synchronized void setElement(int index, double value) {
        if (index < 0) {
            throw new ArrayIndexOutOfBoundsException(index);
        }
        if (index + 1 > numElements) {
            numElements = index + 1;
        }
        if ((startIndex + index) >= internalArray.length) {
            expandTo(startIndex + (index + 1));
        }
        internalArray[startIndex + index] = value;
    }

    
    @Deprecated
    public void setExpansionFactor(float expansionFactor) throws MathIllegalArgumentException {
        checkContractExpand(getContractionCriterion(), expansionFactor);
        // The check above verifies that the expansion factor is > 1.0;
        synchronized(this) {
            this.expansionFactor = expansionFactor;
        }
    }

    
    @Deprecated
    public void setExpansionMode(int expansionMode)
        throws MathIllegalArgumentException {
        if (expansionMode != MULTIPLICATIVE_MODE &&
            expansionMode != ADDITIVE_MODE) {
            throw new MathIllegalArgumentException(LocalizedFormats.UNSUPPORTED_EXPANSION_MODE, expansionMode,
                                                   MULTIPLICATIVE_MODE, "MULTIPLICATIVE_MODE",
                                                   ADDITIVE_MODE, "ADDITIVE_MODE");
        }
        synchronized(this) {
            if (expansionMode == MULTIPLICATIVE_MODE) {
                setExpansionMode(ExpansionMode.MULTIPLICATIVE);
            } else if (expansionMode == ADDITIVE_MODE) {
                setExpansionMode(ExpansionMode.ADDITIVE);
            }
        }
    }

    
    @Deprecated
    public void setExpansionMode(ExpansionMode expansionMode) {
        synchronized(this) {
            this.expansionMode = expansionMode;
        }
    }

    
    @Deprecated
    protected void setInitialCapacity(int initialCapacity)
        throws MathIllegalArgumentException {
        // Body removed in 3.1.
    }

    
    public synchronized void setNumElements(int i)
        throws MathIllegalArgumentException {
        // If index is negative thrown an error.
        if (i < 0) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.INDEX_NOT_POSITIVE,
                    i);
        }

        // Test the new num elements, check to see if the array needs to be
        // expanded to accommodate this new number of elements.
        final int newSize = startIndex + i;
        if (newSize > internalArray.length) {
            expandTo(newSize);
        }

        // Set the new number of elements to new value.
        numElements = i;
    }

    
    private synchronized boolean shouldContract() {
        if (expansionMode == ExpansionMode.MULTIPLICATIVE) {
            return (internalArray.length / ((float) numElements)) > contractionCriterion;
        } else {
            return (internalArray.length - numElements) > contractionCriterion;
        }
    }

    
    @Deprecated
    public synchronized int start() {
        return startIndex;
    }

    
    public static void copy(ResizableDoubleArray source,
                            ResizableDoubleArray dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        synchronized(source) {
           synchronized(dest) {
               dest.contractionCriterion = source.contractionCriterion;
               dest.expansionFactor = source.expansionFactor;
               dest.expansionMode = source.expansionMode;
               dest.internalArray = new double[source.internalArray.length];
               System.arraycopy(source.internalArray, 0, dest.internalArray,
                       0, dest.internalArray.length);
               dest.numElements = source.numElements;
               dest.startIndex = source.startIndex;
           }
       }
    }

    
    public synchronized ResizableDoubleArray copy() {
        final ResizableDoubleArray result = new ResizableDoubleArray();
        copy(this, result);
        return result;
    }

    
    @Override
    public boolean equals(Object object) {
        if (object == this ) {
            return true;
        }
        if (object instanceof ResizableDoubleArray == false) {
            return false;
        }
        synchronized(this) {
            synchronized(object) {
                boolean result = true;
                final ResizableDoubleArray other = (ResizableDoubleArray) object;
                result = result && (other.contractionCriterion == contractionCriterion);
                result = result && (other.expansionFactor == expansionFactor);
                result = result && (other.expansionMode == expansionMode);
                result = result && (other.numElements == numElements);
                result = result && (other.startIndex == startIndex);
                if (!result) {
                    return false;
                } else {
                    return Arrays.equals(internalArray, other.internalArray);
                }
            }
        }
    }

    
    @Override
    public synchronized int hashCode() {
        final int[] hashData = new int[6];
        hashData[0] = Double.valueOf(expansionFactor).hashCode();
        hashData[1] = Double.valueOf(contractionCriterion).hashCode();
        hashData[2] = expansionMode.hashCode();
        hashData[3] = Arrays.hashCode(internalArray);
        hashData[4] = numElements;
        hashData[5] = startIndex;
        return Arrays.hashCode(hashData);
    }

}
