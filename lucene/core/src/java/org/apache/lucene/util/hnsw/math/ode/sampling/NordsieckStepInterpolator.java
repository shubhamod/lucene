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

package org.apache.lucene.util.hnsw.math.ode.sampling;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;

import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.linear.Array2DRowRealMatrix;
import org.apache.lucene.util.hnsw.math.ode.EquationsMapper;
import org.apache.lucene.util.hnsw.math.util.FastMath;



public class NordsieckStepInterpolator extends AbstractStepInterpolator {

    
    private static final long serialVersionUID = -7179861704951334960L;

    
    protected double[] stateVariation;

    
    private double scalingH;

    
    private double referenceTime;

    
    private double[] scaled;

    
    private Array2DRowRealMatrix nordsieck;

    
    public NordsieckStepInterpolator() {
    }

    
    public NordsieckStepInterpolator(final NordsieckStepInterpolator interpolator) {
        super(interpolator);
        scalingH      = interpolator.scalingH;
        referenceTime = interpolator.referenceTime;
        if (interpolator.scaled != null) {
            scaled = interpolator.scaled.clone();
        }
        if (interpolator.nordsieck != null) {
            nordsieck = new Array2DRowRealMatrix(interpolator.nordsieck.getDataRef(), true);
        }
        if (interpolator.stateVariation != null) {
            stateVariation = interpolator.stateVariation.clone();
        }
    }

    
    @Override
    protected StepInterpolator doCopy() {
        return new NordsieckStepInterpolator(this);
    }

    
    @Override
    public void reinitialize(final double[] y, final boolean forward,
                             final EquationsMapper primaryMapper,
                             final EquationsMapper[] secondaryMappers) {
        super.reinitialize(y, forward, primaryMapper, secondaryMappers);
        stateVariation = new double[y.length];
    }

    
    public void reinitialize(final double time, final double stepSize,
                             final double[] scaledDerivative,
                             final Array2DRowRealMatrix nordsieckVector) {
        this.referenceTime = time;
        this.scalingH      = stepSize;
        this.scaled        = scaledDerivative;
        this.nordsieck     = nordsieckVector;

        // make sure the state and derivatives will depend on the new arrays
        setInterpolatedTime(getInterpolatedTime());

    }

    
    public void rescale(final double stepSize) {

        final double ratio = stepSize / scalingH;
        for (int i = 0; i < scaled.length; ++i) {
            scaled[i] *= ratio;
        }

        final double[][] nData = nordsieck.getDataRef();
        double power = ratio;
        for (int i = 0; i < nData.length; ++i) {
            power *= ratio;
            final double[] nDataI = nData[i];
            for (int j = 0; j < nDataI.length; ++j) {
                nDataI[j] *= power;
            }
        }

        scalingH = stepSize;

    }

    
    public double[] getInterpolatedStateVariation() throws MaxCountExceededException {
        // compute and ignore interpolated state
        // to make sure state variation is computed as a side effect
        getInterpolatedState();
        return stateVariation;
    }

    
    @Override
    protected void computeInterpolatedStateAndDerivatives(final double theta, final double oneMinusThetaH) {

        final double x = interpolatedTime - referenceTime;
        final double normalizedAbscissa = x / scalingH;

        Arrays.fill(stateVariation, 0.0);
        Arrays.fill(interpolatedDerivatives, 0.0);

        // apply Taylor formula from high order to low order,
        // for the sake of numerical accuracy
        final double[][] nData = nordsieck.getDataRef();
        for (int i = nData.length - 1; i >= 0; --i) {
            final int order = i + 2;
            final double[] nDataI = nData[i];
            final double power = FastMath.pow(normalizedAbscissa, order);
            for (int j = 0; j < nDataI.length; ++j) {
                final double d = nDataI[j] * power;
                stateVariation[j]          += d;
                interpolatedDerivatives[j] += order * d;
            }
        }

        for (int j = 0; j < currentState.length; ++j) {
            stateVariation[j] += scaled[j] * normalizedAbscissa;
            interpolatedState[j] = currentState[j] + stateVariation[j];
            interpolatedDerivatives[j] =
                (interpolatedDerivatives[j] + scaled[j] * normalizedAbscissa) / x;
        }

    }

    
    @Override
    public void writeExternal(final ObjectOutput out)
        throws IOException {

        // save the state of the base class
        writeBaseExternal(out);

        // save the local attributes
        out.writeDouble(scalingH);
        out.writeDouble(referenceTime);

        final int n = (currentState == null) ? -1 : currentState.length;
        if (scaled == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            for (int j = 0; j < n; ++j) {
                out.writeDouble(scaled[j]);
            }
        }

        if (nordsieck == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            out.writeObject(nordsieck);
        }

        // we don't save state variation, it will be recomputed

    }

    
    @Override
    public void readExternal(final ObjectInput in)
        throws IOException, ClassNotFoundException {

        // read the base class
        final double t = readBaseExternal(in);

        // read the local attributes
        scalingH      = in.readDouble();
        referenceTime = in.readDouble();

        final int n = (currentState == null) ? -1 : currentState.length;
        final boolean hasScaled = in.readBoolean();
        if (hasScaled) {
            scaled = new double[n];
            for (int j = 0; j < n; ++j) {
                scaled[j] = in.readDouble();
            }
        } else {
            scaled = null;
        }

        final boolean hasNordsieck = in.readBoolean();
        if (hasNordsieck) {
            nordsieck = (Array2DRowRealMatrix) in.readObject();
        } else {
            nordsieck = null;
        }

        if (hasScaled && hasNordsieck) {
            // we can now set the interpolated time and state
            stateVariation = new double[n];
            setInterpolatedTime(t);
        } else {
            stateVariation = null;
        }

    }

}
