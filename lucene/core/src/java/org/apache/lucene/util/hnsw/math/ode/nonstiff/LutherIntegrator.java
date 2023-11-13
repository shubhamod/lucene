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

package org.apache.lucene.util.hnsw.math.ode.nonstiff;

import org.apache.lucene.util.hnsw.math.util.FastMath;




public class LutherIntegrator extends RungeKuttaIntegrator {

    
    private static final double Q = FastMath.sqrt(21);

    
    private static final double[] STATIC_C = {
        1.0, 1.0 / 2.0, 2.0 / 3.0, (7.0 - Q) / 14.0, (7.0 + Q) / 14.0, 1.0
    };

    
    private static final double[][] STATIC_A = {
        {                      1.0        },
        {                   3.0 /   8.0,                  1.0 /   8.0  },
        {                   8.0 /   27.0,                 2.0 /   27.0,                  8.0 /   27.0  },
        { (  -21.0 +   9.0 * Q) /  392.0, ( -56.0 +  8.0 * Q) /  392.0, ( 336.0 -  48.0 * Q) /  392.0, (-63.0 +   3.0 * Q) /  392.0 },
        { (-1155.0 - 255.0 * Q) / 1960.0, (-280.0 - 40.0 * Q) / 1960.0, (   0.0 - 320.0 * Q) / 1960.0, ( 63.0 + 363.0 * Q) / 1960.0,   (2352.0 + 392.0 * Q) / 1960.0 },
        { (  330.0 + 105.0 * Q) /  180.0, ( 120.0 +  0.0 * Q) /  180.0, (-200.0 + 280.0 * Q) /  180.0, (126.0 - 189.0 * Q) /  180.0,   (-686.0 - 126.0 * Q) /  180.0,   (490.0 -  70.0 * Q) / 180.0 }
    };

    
    private static final double[] STATIC_B = {
        1.0 / 20.0, 0, 16.0 / 45.0, 0, 49.0 / 180.0, 49.0 / 180.0, 1.0 / 20.0
    };

    
    public LutherIntegrator(final double step) {
        super("Luther", STATIC_C, STATIC_A, STATIC_B, new LutherStepInterpolator(), step);
    }

}
