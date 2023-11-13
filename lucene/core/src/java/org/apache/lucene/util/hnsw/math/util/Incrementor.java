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

import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;


@Deprecated
public class Incrementor {
    
    private int maximalCount;
    
    private int count = 0;
    
    private final MaxCountExceededCallback maxCountCallback;

    
    public Incrementor() {
        this(0);
    }

    
    public Incrementor(int max) {
        this(max,
             new MaxCountExceededCallback() {
                 
                 public void trigger(int max) throws MaxCountExceededException {
                     throw new MaxCountExceededException(max);
                 }
             });
    }

    
    public Incrementor(int max, MaxCountExceededCallback cb)
        throws NullArgumentException {
        if (cb == null){
            throw new NullArgumentException();
        }
        maximalCount = max;
        maxCountCallback = cb;
    }

    
    public void setMaximalCount(int max) {
        maximalCount = max;
    }

    
    public int getMaximalCount() {
        return maximalCount;
    }

    
    public int getCount() {
        return count;
    }

    
    public boolean canIncrement() {
        return count < maximalCount;
    }

    
    public void incrementCount(int value) throws MaxCountExceededException {
        for (int i = 0; i < value; i++) {
            incrementCount();
        }
    }

    
    public void incrementCount() throws MaxCountExceededException {
        if (++count > maximalCount) {
            maxCountCallback.trigger(maximalCount);
        }
    }

    
    public void resetCount() {
        count = 0;
    }

    
    public interface MaxCountExceededCallback {
        
        void trigger(int maximalCount) throws MaxCountExceededException;
    }

    
    public static Incrementor wrap(final IntegerSequence.Incrementor incrementor) {
        return new Incrementor() {

            
            private IntegerSequence.Incrementor delegate;

            {
                // set up matching values at initialization
                delegate = incrementor;
                super.setMaximalCount(delegate.getMaximalCount());
                super.incrementCount(delegate.getCount());
            }

            
            @Override
            public void setMaximalCount(int max) {
                super.setMaximalCount(max);
                delegate = delegate.withMaximalCount(max);
            }

            
            @Override
            public void resetCount() {
                super.resetCount();
                delegate = delegate.withStart(0);
            }

            
            @Override
            public void incrementCount() {
                super.incrementCount();
                delegate.increment();
            }

        };
    }

}
