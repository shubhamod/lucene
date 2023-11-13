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
package org.apache.lucene.util.hnsw.math.ode;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class JacobianMatrices {

    
    private ExpandableStatefulODE efode;

    
    private int index;

    
    private MainStateJacobianProvider jode;

    
    private ParameterizedODE pode;

    
    private int stateDim;

    
    private ParameterConfiguration[] selectedParameters;

    
    private List<ParameterJacobianProvider> jacobianProviders;

    
    private int paramDim;

    
    private boolean dirtyParameter;

    
    private double[] matricesData;

    
    public JacobianMatrices(final FirstOrderDifferentialEquations fode, final double[] hY,
                            final String... parameters)
        throws DimensionMismatchException {
        this(new MainStateJacobianWrapper(fode, hY), parameters);
    }

    
    public JacobianMatrices(final MainStateJacobianProvider jode,
                            final String... parameters) {

        this.efode = null;
        this.index = -1;

        this.jode = jode;
        this.pode = null;

        this.stateDim = jode.getDimension();

        if (parameters == null) {
            selectedParameters = null;
            paramDim = 0;
        } else {
            this.selectedParameters = new ParameterConfiguration[parameters.length];
            for (int i = 0; i < parameters.length; ++i) {
                selectedParameters[i] = new ParameterConfiguration(parameters[i], Double.NaN);
            }
            paramDim = parameters.length;
        }
        this.dirtyParameter = false;

        this.jacobianProviders = new ArrayList<ParameterJacobianProvider>();

        // set the default initial state Jacobian to the identity
        // and the default initial parameters Jacobian to the null matrix
        matricesData = new double[(stateDim + paramDim) * stateDim];
        for (int i = 0; i < stateDim; ++i) {
            matricesData[i * (stateDim + 1)] = 1.0;
        }

    }

    
    public void registerVariationalEquations(final ExpandableStatefulODE expandable)
        throws DimensionMismatchException, MismatchedEquations {

        // safety checks
        final FirstOrderDifferentialEquations ode = (jode instanceof MainStateJacobianWrapper) ?
                                                    ((MainStateJacobianWrapper) jode).ode :
                                                    jode;
        if (expandable.getPrimary() != ode) {
            throw new MismatchedEquations();
        }

        efode = expandable;
        index = efode.addSecondaryEquations(new JacobiansSecondaryEquations());
        efode.setSecondaryState(index, matricesData);

    }

    
    public void addParameterJacobianProvider(final ParameterJacobianProvider provider) {
        jacobianProviders.add(provider);
    }

    
    public void setParameterizedODE(final ParameterizedODE parameterizedOde) {
        this.pode = parameterizedOde;
        dirtyParameter = true;
    }

    
    public void setParameterStep(final String parameter, final double hP)
        throws UnknownParameterException {

        for (ParameterConfiguration param: selectedParameters) {
            if (parameter.equals(param.getParameterName())) {
                param.setHP(hP);
                dirtyParameter = true;
                return;
            }
        }

        throw new UnknownParameterException(parameter);

    }

    
    public void setInitialMainStateJacobian(final double[][] dYdY0)
        throws DimensionMismatchException {

        // Check dimensions
        checkDimension(stateDim, dYdY0);
        checkDimension(stateDim, dYdY0[0]);

        // store the matrix in row major order as a single dimension array
        int i = 0;
        for (final double[] row : dYdY0) {
            System.arraycopy(row, 0, matricesData, i, stateDim);
            i += stateDim;
        }

        if (efode != null) {
            efode.setSecondaryState(index, matricesData);
        }

    }

    
    public void setInitialParameterJacobian(final String pName, final double[] dYdP)
        throws UnknownParameterException, DimensionMismatchException {

        // Check dimensions
        checkDimension(stateDim, dYdP);

        // store the column in a global single dimension array
        int i = stateDim * stateDim;
        for (ParameterConfiguration param: selectedParameters) {
            if (pName.equals(param.getParameterName())) {
                System.arraycopy(dYdP, 0, matricesData, i, stateDim);
                if (efode != null) {
                    efode.setSecondaryState(index, matricesData);
                }
                return;
            }
            i += stateDim;
        }

        throw new UnknownParameterException(pName);

    }

    
    public void getCurrentMainSetJacobian(final double[][] dYdY0) {

        // get current state for this set of equations from the expandable fode
        double[] p = efode.getSecondaryState(index);

        int j = 0;
        for (int i = 0; i < stateDim; i++) {
            System.arraycopy(p, j, dYdY0[i], 0, stateDim);
            j += stateDim;
        }

    }

    
    public void getCurrentParameterJacobian(String pName, final double[] dYdP) {

        // get current state for this set of equations from the expandable fode
        double[] p = efode.getSecondaryState(index);

        int i = stateDim * stateDim;
        for (ParameterConfiguration param: selectedParameters) {
            if (param.getParameterName().equals(pName)) {
                System.arraycopy(p, i, dYdP, 0, stateDim);
                return;
            }
            i += stateDim;
        }

    }

    
    private void checkDimension(final int expected, final Object array)
        throws DimensionMismatchException {
        int arrayDimension = (array == null) ? 0 : Array.getLength(array);
        if (arrayDimension != expected) {
            throw new DimensionMismatchException(arrayDimension, expected);
        }
    }

    
    private class JacobiansSecondaryEquations implements SecondaryEquations {

        
        public int getDimension() {
            return stateDim * (stateDim + paramDim);
        }

        
        public void computeDerivatives(final double t, final double[] y, final double[] yDot,
                                       final double[] z, final double[] zDot)
            throws MaxCountExceededException, DimensionMismatchException {

            // Lazy initialization
            if (dirtyParameter && (paramDim != 0)) {
                jacobianProviders.add(new ParameterJacobianWrapper(jode, pode, selectedParameters));
                dirtyParameter = false;
            }

            // variational equations:
            // from d[dy/dt]/dy0 and d[dy/dt]/dp to d[dy/dy0]/dt and d[dy/dp]/dt

            // compute Jacobian matrix with respect to primary state
            double[][] dFdY = new double[stateDim][stateDim];
            jode.computeMainStateJacobian(t, y, yDot, dFdY);

            // Dispatch Jacobian matrix in the compound secondary state vector
            for (int i = 0; i < stateDim; ++i) {
                final double[] dFdYi = dFdY[i];
                for (int j = 0; j < stateDim; ++j) {
                    double s = 0;
                    final int startIndex = j;
                    int zIndex = startIndex;
                    for (int l = 0; l < stateDim; ++l) {
                        s += dFdYi[l] * z[zIndex];
                        zIndex += stateDim;
                    }
                    zDot[startIndex + i * stateDim] = s;
                }
            }

            if (paramDim != 0) {
                // compute Jacobian matrices with respect to parameters
                double[] dFdP = new double[stateDim];
                int startIndex = stateDim * stateDim;
                for (ParameterConfiguration param: selectedParameters) {
                    boolean found = false;
                    for (int k = 0 ; (!found) && (k < jacobianProviders.size()); ++k) {
                        final ParameterJacobianProvider provider = jacobianProviders.get(k);
                        if (provider.isSupported(param.getParameterName())) {
                            provider.computeParameterJacobian(t, y, yDot,
                                                              param.getParameterName(), dFdP);
                            for (int i = 0; i < stateDim; ++i) {
                                final double[] dFdYi = dFdY[i];
                                int zIndex = startIndex;
                                double s = dFdP[i];
                                for (int l = 0; l < stateDim; ++l) {
                                    s += dFdYi[l] * z[zIndex];
                                    zIndex++;
                                }
                                zDot[startIndex + i] = s;
                            }
                            found = true;
                        }
                    }
                    if (! found) {
                        Arrays.fill(zDot, startIndex, startIndex + stateDim, 0.0);
                    }
                    startIndex += stateDim;
                }
            }

        }
    }

    
    private static class MainStateJacobianWrapper implements MainStateJacobianProvider {

        
        private final FirstOrderDifferentialEquations ode;

        
        private final double[] hY;

        
        MainStateJacobianWrapper(final FirstOrderDifferentialEquations ode,
                                 final double[] hY)
            throws DimensionMismatchException {
            this.ode = ode;
            this.hY = hY.clone();
            if (hY.length != ode.getDimension()) {
                throw new DimensionMismatchException(ode.getDimension(), hY.length);
            }
        }

        
        public int getDimension() {
            return ode.getDimension();
        }

        
        public void computeDerivatives(double t, double[] y, double[] yDot)
            throws MaxCountExceededException, DimensionMismatchException {
            ode.computeDerivatives(t, y, yDot);
        }

        
        public void computeMainStateJacobian(double t, double[] y, double[] yDot, double[][] dFdY)
            throws MaxCountExceededException, DimensionMismatchException {

            final int n = ode.getDimension();
            final double[] tmpDot = new double[n];

            for (int j = 0; j < n; ++j) {
                final double savedYj = y[j];
                y[j] += hY[j];
                ode.computeDerivatives(t, y, tmpDot);
                for (int i = 0; i < n; ++i) {
                    dFdY[i][j] = (tmpDot[i] - yDot[i]) / hY[j];
                }
                y[j] = savedYj;
            }
        }

    }

    
    public static class MismatchedEquations extends MathIllegalArgumentException {

        
        private static final long serialVersionUID = 20120902L;

        
        public MismatchedEquations() {
            super(LocalizedFormats.UNMATCHED_ODE_IN_EXPANDED_SET);
        }

    }

}

