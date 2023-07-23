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



public class Well512a extends AbstractWell {

    
    private static final long serialVersionUID = -6104179812103820574L;

    
    private static final int K = 512;

    
    private static final int M1 = 13;

    
    private static final int M2 = 9;

    
    private static final int M3 = 5;

    
    public Well512a() {
        super(K, M1, M2, M3);
    }

    
    public Well512a(int seed) {
        super(K, M1, M2, M3, seed);
    }

    
    public Well512a(int[] seed) {
        super(K, M1, M2, M3, seed);
    }

    
    public Well512a(long seed) {
        super(K, M1, M2, M3, seed);
    }

    
    @Override
    protected int next(final int bits) {

        final int indexRm1 = iRm1[index];

        final int vi = v[index];
        final int vi1 = v[i1[index]];
        final int vi2 = v[i2[index]];
        final int z0 = v[indexRm1];

        // the values below include the errata of the original article
        final int z1 = (vi ^ (vi << 16))   ^ (vi1 ^ (vi1 << 15));
        final int z2 = vi2 ^ (vi2 >>> 11);
        final int z3 = z1 ^ z2;
        final int z4 = (z0 ^ (z0 << 2)) ^ (z1 ^ (z1 << 18)) ^ (z2 << 28) ^ (z3 ^ ((z3 << 5) & 0xda442d24));

        v[index] = z3;
        v[indexRm1]  = z4;
        index    = indexRm1;

        return z4 >>> (32 - bits);

    }

}
