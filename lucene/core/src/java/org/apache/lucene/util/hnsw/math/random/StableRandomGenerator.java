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

import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class StableRandomGenerator implements NormalizedRandomGenerator {
    
    private final RandomGenerator generator;

    
    private final double alpha;

    
    private final double beta;

    
    private final double zeta;

    
    public StableRandomGenerator(final RandomGenerator generator,
                                 final double alpha, final double beta)
        throws NullArgumentException, OutOfRangeException {
        if (generator == null) {
            throw new NullArgumentException();
        }

        if (!(alpha > 0d && alpha <= 2d)) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_RANGE_LEFT,
                    alpha, 0, 2);
        }

        if (!(beta >= -1d && beta <= 1d)) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_RANGE_SIMPLE,
                    beta, -1, 1);
        }

        this.generator = generator;
        this.alpha = alpha;
        this.beta = beta;
        if (alpha < 2d && beta != 0d) {
            zeta = beta * FastMath.tan(FastMath.PI * alpha / 2);
        } else {
            zeta = 0d;
        }
    }

    
    public double nextNormalizedDouble() {
        // we need 2 uniform random numbers to calculate omega and phi
        double omega = -FastMath.log(generator.nextDouble());
        double phi = FastMath.PI * (generator.nextDouble() - 0.5);

        // Normal distribution case (Box-Muller algorithm)
        if (alpha == 2d) {
            return FastMath.sqrt(2d * omega) * FastMath.sin(phi);
        }

        double x;
        // when beta = 0, zeta is zero as well
        // Thus we can exclude it from the formula
        if (beta == 0d) {
            // Cauchy distribution case
            if (alpha == 1d) {
                x = FastMath.tan(phi);
            } else {
                x = FastMath.pow(omega * FastMath.cos((1 - alpha) * phi),
                    1d / alpha - 1d) *
                    FastMath.sin(alpha * phi) /
                    FastMath.pow(FastMath.cos(phi), 1d / alpha);
            }
        } else {
            // Generic stable distribution
            double cosPhi = FastMath.cos(phi);
            // to avoid rounding errors around alpha = 1
            if (FastMath.abs(alpha - 1d) > 1e-8) {
                double alphaPhi = alpha * phi;
                double invAlphaPhi = phi - alphaPhi;
                x = (FastMath.sin(alphaPhi) + zeta * FastMath.cos(alphaPhi)) / cosPhi *
                    (FastMath.cos(invAlphaPhi) + zeta * FastMath.sin(invAlphaPhi)) /
                     FastMath.pow(omega * cosPhi, (1 - alpha) / alpha);
            } else {
                double betaPhi = FastMath.PI / 2 + beta * phi;
                x = 2d / FastMath.PI * (betaPhi * FastMath.tan(phi) - beta *
                    FastMath.log(FastMath.PI / 2d * omega * cosPhi / betaPhi));

                if (alpha != 1d) {
                    x += beta * FastMath.tan(FastMath.PI * alpha / 2);
                }
            }
        }
        return x;
    }
}
