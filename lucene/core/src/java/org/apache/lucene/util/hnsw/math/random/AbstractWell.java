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

import org.apache.lucene.util.hnsw.math.util.FastMath;



public abstract class AbstractWell extends BitsStreamGenerator implements Serializable {

    
    private static final long serialVersionUID = -817701723016583596L;

    
    protected int index;

    
    protected final int[] v;

    
    protected final int[] iRm1;

    
    protected final int[] iRm2;

    
    protected final int[] i1;

    
    protected final int[] i2;

    
    protected final int[] i3;

    
    protected AbstractWell(final int k, final int m1, final int m2, final int m3) {
        this(k, m1, m2, m3, null);
    }

    
    protected AbstractWell(final int k, final int m1, final int m2, final int m3, final int seed) {
        this(k, m1, m2, m3, new int[] { seed });
    }

    
    protected AbstractWell(final int k, final int m1, final int m2, final int m3, final int[] seed) {

        // the bits pool contains k bits, k = r w - p where r is the number
        // of w bits blocks, w is the block size (always 32 in the original paper)
        // and p is the number of unused bits in the last block
        final int w = 32;
        final int r = (k + w - 1) / w;
        this.v      = new int[r];
        this.index  = 0;

        // precompute indirection index tables. These tables are used for optimizing access
        // they allow saving computations like "(j + r - 2) % r" with costly modulo operations
        iRm1 = new int[r];
        iRm2 = new int[r];
        i1   = new int[r];
        i2   = new int[r];
        i3   = new int[r];
        for (int j = 0; j < r; ++j) {
            iRm1[j] = (j + r - 1) % r;
            iRm2[j] = (j + r - 2) % r;
            i1[j]   = (j + m1)    % r;
            i2[j]   = (j + m2)    % r;
            i3[j]   = (j + m3)    % r;
        }

        // initialize the pool content
        setSeed(seed);

    }

    
    protected AbstractWell(final int k, final int m1, final int m2, final int m3, final long seed) {
        this(k, m1, m2, m3, new int[] { (int) (seed >>> 32), (int) (seed & 0xffffffffl) });
    }

    
    @Override
    public void setSeed(final int seed) {
        setSeed(new int[] { seed });
    }

    
    @Override
    public void setSeed(final int[] seed) {
        if (seed == null) {
            setSeed(System.currentTimeMillis() + System.identityHashCode(this));
            return;
        }

        System.arraycopy(seed, 0, v, 0, FastMath.min(seed.length, v.length));

        if (seed.length < v.length) {
            for (int i = seed.length; i < v.length; ++i) {
                final long l = v[i - seed.length];
                v[i] = (int) ((1812433253l * (l ^ (l >> 30)) + i) & 0xffffffffL);
            }
        }

        index = 0;
        clear();  // Clear normal deviate cache
    }

    
    @Override
    public void setSeed(final long seed) {
        setSeed(new int[] { (int) (seed >>> 32), (int) (seed & 0xffffffffl) });
    }

    
    @Override
    protected abstract int next(final int bits);

}
