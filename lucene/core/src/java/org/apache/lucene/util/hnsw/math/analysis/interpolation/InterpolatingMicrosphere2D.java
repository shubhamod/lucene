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
package org.apache.lucene.util.hnsw.math.analysis.interpolation;

import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class InterpolatingMicrosphere2D extends InterpolatingMicrosphere {
    
    private static final int DIMENSION = 2;

    
    public InterpolatingMicrosphere2D(int size,
                                      double maxDarkFraction,
                                      double darkThreshold,
                                      double background) {
        super(DIMENSION, size, maxDarkFraction, darkThreshold, background);

        // Generate the microsphere normals.
        for (int i = 0; i < size; i++) {
            final double angle = i * MathUtils.TWO_PI / size;

            add(new double[] { FastMath.cos(angle),
                               FastMath.sin(angle) },
                false);
        }
    }

    
    protected InterpolatingMicrosphere2D(InterpolatingMicrosphere2D other) {
        super(other);
    }

    
    @Override
    public InterpolatingMicrosphere2D copy() {
        return new InterpolatingMicrosphere2D(this);
    }
}
