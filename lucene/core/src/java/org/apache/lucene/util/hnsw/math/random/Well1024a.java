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



public class Well1024a extends AbstractWell {

    
    private static final long serialVersionUID = 5680173464174485492L;

    
    private static final int K = 1024;

    
    private static final int M1 = 3;

    
    private static final int M2 = 24;

    
    private static final int M3 = 10;

    
    public Well1024a() {
        super(K, M1, M2, M3);
    }

    
    public Well1024a(int seed) {
        super(K, M1, M2, M3, seed);
    }

    
    public Well1024a(int[] seed) {
        super(K, M1, M2, M3, seed);
    }

    
    public Well1024a(long seed) {
        super(K, M1, M2, M3, seed);
    }

    
    @Override
    protected int next(final int bits) {

        final int indexRm1 = iRm1[index];

        final int v0       = v[index];
        final int vM1      = v[i1[index]];
        final int vM2      = v[i2[index]];
        final int vM3      = v[i3[index]];

        final int z0 = v[indexRm1];
        final int z1 = v0  ^ (vM1 ^ (vM1 >>> 8));
        final int z2 = (vM2 ^ (vM2 << 19)) ^ (vM3 ^ (vM3 << 14));
        final int z3 = z1      ^ z2;
        final int z4 = (z0 ^ (z0 << 11)) ^ (z1 ^ (z1 << 7)) ^ (z2 ^ (z2 << 13));

        v[index]     = z3;
        v[indexRm1]  = z4;
        index        = indexRm1;

        return z4 >>> (32 - bits);

    }
}
