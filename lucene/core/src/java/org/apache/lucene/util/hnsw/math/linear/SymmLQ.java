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
package org.apache.lucene.util.hnsw.math.linear;

import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.ExceptionContext;
import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.IterationManager;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class SymmLQ
    extends PreconditionedIterativeLinearSolver {

    /*
     * IMPLEMENTATION NOTES
     * --------------------
     * The implementation follows as closely as possible the notations of Paige
     * and Saunders (1975). Attention must be paid to the fact that some
     * quantities which are relevant to iteration k can only be computed in
     * iteration (k+1). Therefore, minute attention must be paid to the index of
     * each state variable of this algorithm.
     *
     * 1. Preconditioning
     *    ---------------
     * The Lanczos iterations associated with Ahat and bhat read
     *   beta[1] = ||P * b||
     *   v[1] = P * b / beta[1]
     *   beta[k+1] * v[k+1] = Ahat * v[k] - alpha[k] * v[k] - beta[k] * v[k-1]
     *                      = P * (A - shift * I) * P' * v[k] - alpha[k] * v[k]
     *                        - beta[k] * v[k-1]
     * Multiplying both sides by P', we get
     *   beta[k+1] * (P' * v)[k+1] = M * (A - shift * I) * (P' * v)[k]
     *                               - alpha[k] * (P' * v)[k]
     *                               - beta[k] * (P' * v[k-1]),
     * and
     *   alpha[k+1] = v[k+1]' * Ahat * v[k+1]
     *              = v[k+1]' * P * (A - shift * I) * P' * v[k+1]
     *              = (P' * v)[k+1]' * (A - shift * I) * (P' * v)[k+1].
     *
     * In other words, the Lanczos iterations are unchanged, except for the fact
     * that we really compute (P' * v) instead of v. It can easily be checked
     * that all other formulas are unchanged. It must be noted that P is never
     * explicitly used, only matrix-vector products involving are invoked.
     *
     * 2. Accounting for the shift parameter
     *    ----------------------------------
     * Is trivial: each time A.operate(x) is invoked, one must subtract shift * x
     * to the result.
     *
     * 3. Accounting for the goodb flag
     *    -----------------------------
     * When goodb is set to true, the component of xL along b is computed
     * separately. From Paige and Saunders (1975), equation (5.9), we have
     *   wbar[k+1] = s[k] * wbar[k] - c[k] * v[k+1],
     *   wbar[1] = v[1].
     * Introducing wbar2[k] = wbar[k] - s[1] * ... * s[k-1] * v[1], it can
     * easily be verified by induction that wbar2 follows the same recursive
     * relation
     *   wbar2[k+1] = s[k] * wbar2[k] - c[k] * v[k+1],
     *   wbar2[1] = 0,
     * and we then have
     *   w[k] = c[k] * wbar2[k] + s[k] * v[k+1]
     *          + s[1] * ... * s[k-1] * c[k] * v[1].
     * Introducing w2[k] = w[k] - s[1] * ... * s[k-1] * c[k] * v[1], we find,
     * from (5.10)
     *   xL[k] = zeta[1] * w[1] + ... + zeta[k] * w[k]
     *         = zeta[1] * w2[1] + ... + zeta[k] * w2[k]
     *           + (s[1] * c[2] * zeta[2] + ...
     *           + s[1] * ... * s[k-1] * c[k] * zeta[k]) * v[1]
     *         = xL2[k] + bstep[k] * v[1],
     * where xL2[k] is defined by
     *   xL2[0] = 0,
     *   xL2[k+1] = xL2[k] + zeta[k+1] * w2[k+1],
     * and bstep is defined by
     *   bstep[1] = 0,
     *   bstep[k] = bstep[k-1] + s[1] * ... * s[k-1] * c[k] * zeta[k].
     * We also have, from (5.11)
     *   xC[k] = xL[k-1] + zbar[k] * wbar[k]
     *         = xL2[k-1] + zbar[k] * wbar2[k]
     *           + (bstep[k-1] + s[1] * ... * s[k-1] * zbar[k]) * v[1].
     */

    
    private static class State {
        
        static final double CBRT_MACH_PREC;

        
        static final double MACH_PREC;

        
        private final RealLinearOperator a;

        
        private final RealVector b;

        
        private final boolean check;

        
        private final double delta;

        
        private double beta;

        
        private double beta1;

        
        private double bstep;

        
        private double cgnorm;

        
        private double dbar;

        
        private double gammaZeta;

        
        private double gbar;

        
        private double gmax;

        
        private double gmin;

        
        private final boolean goodb;

        
        private boolean hasConverged;

        
        private double lqnorm;

        
        private final RealLinearOperator m;

        
        private double minusEpsZeta;

        
        private final RealVector mb;

        
        private double oldb;

        
        private RealVector r1;

        
        private RealVector r2;

        
        private double rnorm;

        
        private final double shift;

        
        private double snprod;

        
        private double tnorm;

        
        private RealVector wbar;

        
        private final RealVector xL;

        
        private RealVector y;

        
        private double ynorm2;

        
        private boolean bIsNull;

        static {
            MACH_PREC = FastMath.ulp(1.);
            CBRT_MACH_PREC = FastMath.cbrt(MACH_PREC);
        }

        
        State(final RealLinearOperator a,
            final RealLinearOperator m,
            final RealVector b,
            final boolean goodb,
            final double shift,
            final double delta,
            final boolean check) {
            this.a = a;
            this.m = m;
            this.b = b;
            this.xL = new ArrayRealVector(b.getDimension());
            this.goodb = goodb;
            this.shift = shift;
            this.mb = m == null ? b : m.operate(b);
            this.hasConverged = false;
            this.check = check;
            this.delta = delta;
        }

        
        private static void checkSymmetry(final RealLinearOperator l,
            final RealVector x, final RealVector y, final RealVector z)
            throws NonSelfAdjointOperatorException {
            final double s = y.dotProduct(y);
            final double t = x.dotProduct(z);
            final double epsa = (s + MACH_PREC) * CBRT_MACH_PREC;
            if (FastMath.abs(s - t) > epsa) {
                final NonSelfAdjointOperatorException e;
                e = new NonSelfAdjointOperatorException();
                final ExceptionContext context = e.getContext();
                context.setValue(SymmLQ.OPERATOR, l);
                context.setValue(SymmLQ.VECTOR1, x);
                context.setValue(SymmLQ.VECTOR2, y);
                context.setValue(SymmLQ.THRESHOLD, Double.valueOf(epsa));
                throw e;
            }
        }

        
        private static void throwNPDLOException(final RealLinearOperator l,
            final RealVector v) throws NonPositiveDefiniteOperatorException {
            final NonPositiveDefiniteOperatorException e;
            e = new NonPositiveDefiniteOperatorException();
            final ExceptionContext context = e.getContext();
            context.setValue(OPERATOR, l);
            context.setValue(VECTOR, v);
            throw e;
        }

        
        private static void daxpy(final double a, final RealVector x,
            final RealVector y) {
            final int n = x.getDimension();
            for (int i = 0; i < n; i++) {
                y.setEntry(i, a * x.getEntry(i) + y.getEntry(i));
            }
        }

        
        private static void daxpbypz(final double a, final RealVector x,
            final double b, final RealVector y, final RealVector z) {
            final int n = z.getDimension();
            for (int i = 0; i < n; i++) {
                final double zi;
                zi = a * x.getEntry(i) + b * y.getEntry(i) + z.getEntry(i);
                z.setEntry(i, zi);
            }
        }

        
         void refineSolution(final RealVector x) {
            final int n = this.xL.getDimension();
            if (lqnorm < cgnorm) {
                if (!goodb) {
                    x.setSubVector(0, this.xL);
                } else {
                    final double step = bstep / beta1;
                    for (int i = 0; i < n; i++) {
                        final double bi = mb.getEntry(i);
                        final double xi = this.xL.getEntry(i);
                        x.setEntry(i, xi + step * bi);
                    }
                }
            } else {
                final double anorm = FastMath.sqrt(tnorm);
                final double diag = gbar == 0. ? anorm * MACH_PREC : gbar;
                final double zbar = gammaZeta / diag;
                final double step = (bstep + snprod * zbar) / beta1;
                // ynorm = FastMath.sqrt(ynorm2 + zbar * zbar);
                if (!goodb) {
                    for (int i = 0; i < n; i++) {
                        final double xi = this.xL.getEntry(i);
                        final double wi = wbar.getEntry(i);
                        x.setEntry(i, xi + zbar * wi);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        final double xi = this.xL.getEntry(i);
                        final double wi = wbar.getEntry(i);
                        final double bi = mb.getEntry(i);
                        x.setEntry(i, xi + zbar * wi + step * bi);
                    }
                }
            }
        }

        
         void init() {
            this.xL.set(0.);
            /*
             * Set up y for the first Lanczos vector. y and beta1 will be zero
             * if b = 0.
             */
            this.r1 = this.b.copy();
            this.y = this.m == null ? this.b.copy() : this.m.operate(this.r1);
            if ((this.m != null) && this.check) {
                checkSymmetry(this.m, this.r1, this.y, this.m.operate(this.y));
            }

            this.beta1 = this.r1.dotProduct(this.y);
            if (this.beta1 < 0.) {
                throwNPDLOException(this.m, this.y);
            }
            if (this.beta1 == 0.) {
                /* If b = 0 exactly, stop with x = 0. */
                this.bIsNull = true;
                return;
            }
            this.bIsNull = false;
            this.beta1 = FastMath.sqrt(this.beta1);
            /* At this point
             *   r1 = b,
             *   y = M * b,
             *   beta1 = beta[1].
             */
            final RealVector v = this.y.mapMultiply(1. / this.beta1);
            this.y = this.a.operate(v);
            if (this.check) {
                checkSymmetry(this.a, v, this.y, this.a.operate(this.y));
            }
            /*
             * Set up y for the second Lanczos vector. y and beta will be zero
             * or very small if b is an eigenvector.
             */
            daxpy(-this.shift, v, this.y);
            final double alpha = v.dotProduct(this.y);
            daxpy(-alpha / this.beta1, this.r1, this.y);
            /*
             * At this point
             *   alpha = alpha[1]
             *   y     = beta[2] * M^(-1) * P' * v[2]
             */
            /* Make sure r2 will be orthogonal to the first v. */
            final double vty = v.dotProduct(this.y);
            final double vtv = v.dotProduct(v);
            daxpy(-vty / vtv, v, this.y);
            this.r2 = this.y.copy();
            if (this.m != null) {
                this.y = this.m.operate(this.r2);
            }
            this.oldb = this.beta1;
            this.beta = this.r2.dotProduct(this.y);
            if (this.beta < 0.) {
                throwNPDLOException(this.m, this.y);
            }
            this.beta = FastMath.sqrt(this.beta);
            /*
             * At this point
             *   oldb = beta[1]
             *   beta = beta[2]
             *   y  = beta[2] * P' * v[2]
             *   r2 = beta[2] * M^(-1) * P' * v[2]
             */
            this.cgnorm = this.beta1;
            this.gbar = alpha;
            this.dbar = this.beta;
            this.gammaZeta = this.beta1;
            this.minusEpsZeta = 0.;
            this.bstep = 0.;
            this.snprod = 1.;
            this.tnorm = alpha * alpha + this.beta * this.beta;
            this.ynorm2 = 0.;
            this.gmax = FastMath.abs(alpha) + MACH_PREC;
            this.gmin = this.gmax;

            if (this.goodb) {
                this.wbar = new ArrayRealVector(this.a.getRowDimension());
                this.wbar.set(0.);
            } else {
                this.wbar = v;
            }
            updateNorms();
        }

        
        void update() {
            final RealVector v = y.mapMultiply(1. / beta);
            y = a.operate(v);
            daxpbypz(-shift, v, -beta / oldb, r1, y);
            final double alpha = v.dotProduct(y);
            /*
             * At this point
             *   v     = P' * v[k],
             *   y     = (A - shift * I) * P' * v[k] - beta[k] * M^(-1) * P' * v[k-1],
             *   alpha = v'[k] * P * (A - shift * I) * P' * v[k]
             *           - beta[k] * v[k]' * P * M^(-1) * P' * v[k-1]
             *         = v'[k] * P * (A - shift * I) * P' * v[k]
             *           - beta[k] * v[k]' * v[k-1]
             *         = alpha[k].
             */
            daxpy(-alpha / beta, r2, y);
            /*
             * At this point
             *   y = (A - shift * I) * P' * v[k] - alpha[k] * M^(-1) * P' * v[k]
             *       - beta[k] * M^(-1) * P' * v[k-1]
             *     = M^(-1) * P' * (P * (A - shift * I) * P' * v[k] -alpha[k] * v[k]
             *       - beta[k] * v[k-1])
             *     = beta[k+1] * M^(-1) * P' * v[k+1],
             * from Paige and Saunders (1975), equation (3.2).
             *
             * WATCH-IT: the two following lines work only because y is no longer
             * updated up to the end of the present iteration, and is
             * reinitialized at the beginning of the next iteration.
             */
            r1 = r2;
            r2 = y;
            if (m != null) {
                y = m.operate(r2);
            }
            oldb = beta;
            beta = r2.dotProduct(y);
            if (beta < 0.) {
                throwNPDLOException(m, y);
            }
            beta = FastMath.sqrt(beta);
            /*
             * At this point
             *   r1 = beta[k] * M^(-1) * P' * v[k],
             *   r2 = beta[k+1] * M^(-1) * P' * v[k+1],
             *   y  = beta[k+1] * P' * v[k+1],
             *   oldb = beta[k],
             *   beta = beta[k+1].
             */
            tnorm += alpha * alpha + oldb * oldb + beta * beta;
            /*
             * Compute the next plane rotation for Q. See Paige and Saunders
             * (1975), equation (5.6), with
             *   gamma = gamma[k-1],
             *   c     = c[k-1],
             *   s     = s[k-1].
             */
            final double gamma = FastMath.sqrt(gbar * gbar + oldb * oldb);
            final double c = gbar / gamma;
            final double s = oldb / gamma;
            /*
             * The relations
             *   gbar[k] = s[k-1] * (-c[k-2] * beta[k]) - c[k-1] * alpha[k]
             *           = s[k-1] * dbar[k] - c[k-1] * alpha[k],
             *   delta[k] = c[k-1] * dbar[k] + s[k-1] * alpha[k],
             * are not stated in Paige and Saunders (1975), but can be retrieved
             * by expanding the (k, k-1) and (k, k) coefficients of the matrix in
             * equation (5.5).
             */
            final double deltak = c * dbar + s * alpha;
            gbar = s * dbar - c * alpha;
            final double eps = s * beta;
            dbar = -c * beta;
            final double zeta = gammaZeta / gamma;
            /*
             * At this point
             *   gbar   = gbar[k]
             *   deltak = delta[k]
             *   eps    = eps[k+1]
             *   dbar   = dbar[k+1]
             *   zeta   = zeta[k-1]
             */
            final double zetaC = zeta * c;
            final double zetaS = zeta * s;
            final int n = xL.getDimension();
            for (int i = 0; i < n; i++) {
                final double xi = xL.getEntry(i);
                final double vi = v.getEntry(i);
                final double wi = wbar.getEntry(i);
                xL.setEntry(i, xi + wi * zetaC + vi * zetaS);
                wbar.setEntry(i, wi * s - vi * c);
            }
            /*
             * At this point
             *   x = xL[k-1],
             *   ptwbar = P' wbar[k],
             * see Paige and Saunders (1975), equations (5.9) and (5.10).
             */
            bstep += snprod * c * zeta;
            snprod *= s;
            gmax = FastMath.max(gmax, gamma);
            gmin = FastMath.min(gmin, gamma);
            ynorm2 += zeta * zeta;
            gammaZeta = minusEpsZeta - deltak * zeta;
            minusEpsZeta = -eps * zeta;
            /*
             * At this point
             *   snprod       = s[1] * ... * s[k-1],
             *   gmax         = max(|alpha[1]|, gamma[1], ..., gamma[k-1]),
             *   gmin         = min(|alpha[1]|, gamma[1], ..., gamma[k-1]),
             *   ynorm2       = zeta[1]^2 + ... + zeta[k-1]^2,
             *   gammaZeta    = gamma[k] * zeta[k],
             *   minusEpsZeta = -eps[k+1] * zeta[k-1].
             * The relation for gammaZeta can be retrieved from Paige and
             * Saunders (1975), equation (5.4a), last line of the vector
             * gbar[k] * zbar[k] = -eps[k] * zeta[k-2] - delta[k] * zeta[k-1].
             */
            updateNorms();
        }

        
        private void updateNorms() {
            final double anorm = FastMath.sqrt(tnorm);
            final double ynorm = FastMath.sqrt(ynorm2);
            final double epsa = anorm * MACH_PREC;
            final double epsx = anorm * ynorm * MACH_PREC;
            final double epsr = anorm * ynorm * delta;
            final double diag = gbar == 0. ? epsa : gbar;
            lqnorm = FastMath.sqrt(gammaZeta * gammaZeta +
                                   minusEpsZeta * minusEpsZeta);
            final double qrnorm = snprod * beta1;
            cgnorm = qrnorm * beta / FastMath.abs(diag);

            /*
             * Estimate cond(A). In this version we look at the diagonals of L
             * in the factorization of the tridiagonal matrix, T = L * Q.
             * Sometimes, T[k] can be misleadingly ill-conditioned when T[k+1]
             * is not, so we must be careful not to overestimate acond.
             */
            final double acond;
            if (lqnorm <= cgnorm) {
                acond = gmax / gmin;
            } else {
                acond = gmax / FastMath.min(gmin, FastMath.abs(diag));
            }
            if (acond * MACH_PREC >= 0.1) {
                throw new IllConditionedOperatorException(acond);
            }
            if (beta1 <= epsx) {
                /*
                 * x has converged to an eigenvector of A corresponding to the
                 * eigenvalue shift.
                 */
                throw new SingularOperatorException();
            }
            rnorm = FastMath.min(cgnorm, lqnorm);
            hasConverged = (cgnorm <= epsx) || (cgnorm <= epsr);
        }

        
        boolean hasConverged() {
            return hasConverged;
        }

        
        boolean bEqualsNullVector() {
            return bIsNull;
        }

        
        boolean betaEqualsZero() {
            return beta < MACH_PREC;
        }

        
        double getNormOfResidual() {
            return rnorm;
        }
    }

    
    private static final String OPERATOR = "operator";

    
    private static final String THRESHOLD = "threshold";

    
    private static final String VECTOR = "vector";

    
    private static final String VECTOR1 = "vector1";

    
    private static final String VECTOR2 = "vector2";

    
    private final boolean check;

    
    private final double delta;

    
    public SymmLQ(final int maxIterations, final double delta,
                  final boolean check) {
        super(maxIterations);
        this.delta = delta;
        this.check = check;
    }

    
    public SymmLQ(final IterationManager manager, final double delta,
                  final boolean check) {
        super(manager);
        this.delta = delta;
        this.check = check;
    }

    
    public final boolean getCheck() {
        return check;
    }

    
    @Override
    public RealVector solve(final RealLinearOperator a,
        final RealLinearOperator m, final RealVector b) throws
        NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, MaxCountExceededException,
        NonSelfAdjointOperatorException, NonPositiveDefiniteOperatorException,
        IllConditionedOperatorException {
        MathUtils.checkNotNull(a);
        final RealVector x = new ArrayRealVector(a.getColumnDimension());
        return solveInPlace(a, m, b, x, false, 0.);
    }

    
    public RealVector solve(final RealLinearOperator a,
        final RealLinearOperator m, final RealVector b, final boolean goodb,
        final double shift) throws NullArgumentException,
        NonSquareOperatorException, DimensionMismatchException,
        MaxCountExceededException, NonSelfAdjointOperatorException,
        NonPositiveDefiniteOperatorException, IllConditionedOperatorException {
        MathUtils.checkNotNull(a);
        final RealVector x = new ArrayRealVector(a.getColumnDimension());
        return solveInPlace(a, m, b, x, goodb, shift);
    }

    
    @Override
    public RealVector solve(final RealLinearOperator a,
        final RealLinearOperator m, final RealVector b, final RealVector x)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, NonSelfAdjointOperatorException,
        NonPositiveDefiniteOperatorException, IllConditionedOperatorException,
        MaxCountExceededException {
        MathUtils.checkNotNull(x);
        return solveInPlace(a, m, b, x.copy(), false, 0.);
    }

    
    @Override
    public RealVector solve(final RealLinearOperator a, final RealVector b)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, NonSelfAdjointOperatorException,
        IllConditionedOperatorException, MaxCountExceededException {
        MathUtils.checkNotNull(a);
        final RealVector x = new ArrayRealVector(a.getColumnDimension());
        x.set(0.);
        return solveInPlace(a, null, b, x, false, 0.);
    }

    
    public RealVector solve(final RealLinearOperator a, final RealVector b,
        final boolean goodb, final double shift) throws NullArgumentException,
        NonSquareOperatorException, DimensionMismatchException,
        NonSelfAdjointOperatorException, IllConditionedOperatorException,
        MaxCountExceededException {
        MathUtils.checkNotNull(a);
        final RealVector x = new ArrayRealVector(a.getColumnDimension());
        return solveInPlace(a, null, b, x, goodb, shift);
    }

    
    @Override
    public RealVector solve(final RealLinearOperator a, final RealVector b,
        final RealVector x) throws NullArgumentException,
        NonSquareOperatorException, DimensionMismatchException,
        NonSelfAdjointOperatorException, IllConditionedOperatorException,
        MaxCountExceededException {
        MathUtils.checkNotNull(x);
        return solveInPlace(a, null, b, x.copy(), false, 0.);
    }

    
    @Override
    public RealVector solveInPlace(final RealLinearOperator a,
        final RealLinearOperator m, final RealVector b, final RealVector x)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, NonSelfAdjointOperatorException,
        NonPositiveDefiniteOperatorException, IllConditionedOperatorException,
        MaxCountExceededException {
        return solveInPlace(a, m, b, x, false, 0.);
    }

    
    public RealVector solveInPlace(final RealLinearOperator a,
        final RealLinearOperator m, final RealVector b,
        final RealVector x, final boolean goodb, final double shift)
        throws NullArgumentException, NonSquareOperatorException,
        DimensionMismatchException, NonSelfAdjointOperatorException,
        NonPositiveDefiniteOperatorException, IllConditionedOperatorException,
        MaxCountExceededException {
        checkParameters(a, m, b, x);

        final IterationManager manager = getIterationManager();
        /* Initialization counts as an iteration. */
        manager.resetIterationCount();
        manager.incrementIterationCount();

        final State state;
        state = new State(a, m, b, goodb, shift, delta, check);
        state.init();
        state.refineSolution(x);
        IterativeLinearSolverEvent event;
        event = new DefaultIterativeLinearSolverEvent(this,
                                                      manager.getIterations(),
                                                      x,
                                                      b,
                                                      state.getNormOfResidual());
        if (state.bEqualsNullVector()) {
            /* If b = 0 exactly, stop with x = 0. */
            manager.fireTerminationEvent(event);
            return x;
        }
        /* Cause termination if beta is essentially zero. */
        final boolean earlyStop;
        earlyStop = state.betaEqualsZero() || state.hasConverged();
        manager.fireInitializationEvent(event);
        if (!earlyStop) {
            do {
                manager.incrementIterationCount();
                event = new DefaultIterativeLinearSolverEvent(this,
                                                              manager.getIterations(),
                                                              x,
                                                              b,
                                                              state.getNormOfResidual());
                manager.fireIterationStartedEvent(event);
                state.update();
                state.refineSolution(x);
                event = new DefaultIterativeLinearSolverEvent(this,
                                                              manager.getIterations(),
                                                              x,
                                                              b,
                                                              state.getNormOfResidual());
                manager.fireIterationPerformedEvent(event);
            } while (!state.hasConverged());
        }
        event = new DefaultIterativeLinearSolverEvent(this,
                                                      manager.getIterations(),
                                                      x,
                                                      b,
                                                      state.getNormOfResidual());
        manager.fireTerminationEvent(event);
        return x;
    }

    
    @Override
    public RealVector solveInPlace(final RealLinearOperator a,
        final RealVector b, final RealVector x) throws NullArgumentException,
        NonSquareOperatorException, DimensionMismatchException,
        NonSelfAdjointOperatorException, IllConditionedOperatorException,
        MaxCountExceededException {
        return solveInPlace(a, null, b, x, false, 0.);
    }
}
