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



public class Well44497a extends AbstractWell {

    
    private static final long serialVersionUID = -3859207588353972099L;

    
    private static final int K = 44497;

    
    private static final int M1 = 23;

    
    private static final int M2 = 481;

    
    private static final int M3 = 229;

    
    public Well44497a() {
        super(K, M1, M2, M3);
    }

    
    public Well44497a(int seed) {
        super(K, M1, M2, M3, seed);
    }

    
    public Well44497a(int[] seed) {
        super(K, M1, M2, M3, seed);
    }

    
    public Well44497a(long seed) {
        super(K, M1, M2, M3, seed);
    }

    
    @Override
    protected int next(final int bits) {

        final int indexRm1 = iRm1[index];
        final int indexRm2 = iRm2[index];

        final int v0       = v[index];
        final int vM1      = v[i1[index]];
        final int vM2      = v[i2[index]];
        final int vM3      = v[i3[index]];

        // the values below include the errata of the original article
        final int z0       = (0xFFFF8000 & v[indexRm1]) ^ (0x00007FFF & v[indexRm2]);
        final int z1       = (v0 ^ (v0 << 24))  ^ (vM1 ^ (vM1 >>> 30));
        final int z2       = (vM2 ^ (vM2 << 10)) ^ (vM3 << 26);
        final int z3       = z1      ^ z2;
        final int z2Prime  = ((z2 << 9) ^ (z2 >>> 23)) & 0xfbffffff;
        final int z2Second = ((z2 & 0x00020000) != 0) ? (z2Prime ^ 0xb729fcec) : z2Prime;
        final int z4       = z0 ^ (z1 ^ (z1 >>> 20)) ^ z2Second ^ z3;

        v[index]     = z3;
        v[indexRm1]  = z4;
        v[indexRm2] &= 0xFFFF8000;
        index        = indexRm1;

        return z4 >>> (32 - bits);

    }
}
