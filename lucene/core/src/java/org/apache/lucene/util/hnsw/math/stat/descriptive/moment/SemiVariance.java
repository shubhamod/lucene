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

package org.apache.lucene.util.hnsw.math.stat.descriptive.moment;

import java.io.Serializable;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class SemiVariance extends AbstractUnivariateStatistic implements Serializable {

    
    public static final Direction UPSIDE_VARIANCE = Direction.UPSIDE;

    
    public static final Direction DOWNSIDE_VARIANCE = Direction.DOWNSIDE;

    
    private static final long serialVersionUID = -2653430366886024994L;

    
    private boolean biasCorrected = true;

    
    private Direction varianceDirection = Direction.DOWNSIDE;

    
    public SemiVariance() {
    }

    
    public SemiVariance(final boolean biasCorrected) {
        this.biasCorrected = biasCorrected;
    }


    
    public SemiVariance(final Direction direction) {
        this.varianceDirection = direction;
    }


    
    public SemiVariance(final boolean corrected, final Direction direction) {
        this.biasCorrected = corrected;
        this.varianceDirection = direction;
    }


    
    public SemiVariance(final SemiVariance original) throws NullArgumentException {
        copy(original, this);
    }


    
    @Override
    public SemiVariance copy() {
        SemiVariance result = new SemiVariance();
        // No try-catch or advertised exception because args are guaranteed non-null
        copy(this, result);
        return result;
    }


    
    public static void copy(final SemiVariance source, SemiVariance dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        dest.biasCorrected = source.biasCorrected;
        dest.varianceDirection = source.varianceDirection;
    }

    
      @Override
      public double evaluate(final double[] values, final int start, final int length)
      throws MathIllegalArgumentException {
        double m = (new Mean()).evaluate(values, start, length);
        return evaluate(values, m, varianceDirection, biasCorrected, 0, values.length);
      }


      
      public double evaluate(final double[] values, Direction direction)
      throws MathIllegalArgumentException {
          double m = (new Mean()).evaluate(values);
          return evaluate (values, m, direction, biasCorrected, 0, values.length);
      }

      
      public double evaluate(final double[] values, final double cutoff)
      throws MathIllegalArgumentException {
          return evaluate(values, cutoff, varianceDirection, biasCorrected, 0, values.length);
      }

      
      public double evaluate(final double[] values, final double cutoff, final Direction direction)
      throws MathIllegalArgumentException {
          return evaluate(values, cutoff, direction, biasCorrected, 0, values.length);
      }


     
    public double evaluate (final double[] values, final double cutoff, final Direction direction,
            final boolean corrected, final int start, final int length) throws MathIllegalArgumentException {

        test(values, start, length);
        if (values.length == 0) {
            return Double.NaN;
        } else {
            if (values.length == 1) {
                return 0.0;
            } else {
                final boolean booleanDirection = direction.getDirection();

                double dev = 0.0;
                double sumsq = 0.0;
                for (int i = start; i < length; i++) {
                    if ((values[i] > cutoff) == booleanDirection) {
                       dev = values[i] - cutoff;
                       sumsq += dev * dev;
                    }
                }

                if (corrected) {
                    return sumsq / (length - 1.0);
                } else {
                    return sumsq / length;
                }
            }
        }
    }

    
    public boolean isBiasCorrected() {
        return biasCorrected;
    }

    
    public void setBiasCorrected(boolean biasCorrected) {
        this.biasCorrected = biasCorrected;
    }

    
    public Direction getVarianceDirection () {
        return varianceDirection;
    }

    
    public void setVarianceDirection(Direction varianceDirection) {
        this.varianceDirection = varianceDirection;
    }

    
    public enum Direction {
        
        UPSIDE (true),

        
        DOWNSIDE (false);

        
        private boolean direction;

        
        Direction (boolean b) {
            direction = b;
        }

        
        boolean getDirection () {
            return direction;
        }
    }
}
