//! Deterministic sparse SPD helpers for implicit heat conduction.
//!
//! CSR storage with sorted columns; CG matvec in fixed index order; optional
//! Thomas algorithm for tridiagonal 1-D rods. No `HashMap`, no `mul_add`.

use pe_math::libm_ops;

/// Compressed sparse row matrix (stores both triangles explicitly).
#[derive(Clone, Debug, Default)]
pub struct CsrMatrix {
    /// Dimension.
    pub n: usize,
    /// Row pointers, length `n + 1`.
    pub row_ptr: Vec<usize>,
    /// Column indices (sorted ascending within each row).
    pub col_idx: Vec<usize>,
    /// Nonzeros parallel to `col_idx`.
    pub vals: Vec<f64>,
}

impl CsrMatrix {
    /// Build from unsorted COO triples `(row, col, val)` by sorting and summing
    /// duplicates. Deterministic: sorts by `(row, col)`.
    pub fn from_coo(n: usize, rows: &[usize], cols: &[usize], vals: &[f64]) -> Self {
        assert_eq!(rows.len(), cols.len());
        assert_eq!(rows.len(), vals.len());
        let mut triples: Vec<(usize, usize, f64)> = rows
            .iter()
            .zip(cols.iter())
            .zip(vals.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect();
        triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut col_idx = Vec::with_capacity(triples.len());
        let mut out_vals = Vec::with_capacity(triples.len());
        let mut row_counts = vec![0usize; n];
        let mut i = 0usize;
        while i < triples.len() {
            let (r, c, mut v) = triples[i];
            i += 1;
            while i < triples.len() && triples[i].0 == r && triples[i].1 == c {
                v += triples[i].2;
                i += 1;
            }
            debug_assert!(r < n && c < n);
            col_idx.push(c);
            out_vals.push(v);
            row_counts[r] += 1;
        }
        let mut row_ptr = vec![0usize; n + 1];
        for r in 0..n {
            row_ptr[r + 1] = row_ptr[r] + row_counts[r];
        }
        Self {
            n,
            row_ptr,
            col_idx,
            vals: out_vals,
        }
    }

    /// `y = A x` (overwrites `y`).
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);
        for i in 0..self.n {
            let mut s = 0.0;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                s += self.vals[k] * x[self.col_idx[k]];
            }
            y[i] = s;
        }
    }
}

/// Dense vector axpy: `y += a * x` (no FMA intrinsic).
#[inline]
pub fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..x.len() {
        y[i] += a * x[i];
    }
}

/// Dot product (deterministic left-to-right sum).
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Conjugate gradient for SPD `A x = b`.
///
/// When `fixed_iters` is `Some(k)`, always runs exactly `k` iterations (bit-stable
/// across platforms). When `None`, stops at relative residual `tol` or `max_iter`.
pub fn cg_solve(
    a: &CsrMatrix,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iter: usize,
    fixed_iters: Option<usize>,
) -> bool {
    let n = a.n;
    if n == 0 {
        return true;
    }
    let mut r = vec![0.0; n];
    let mut ap = vec![0.0; n];
    a.matvec(x, &mut ap);
    for i in 0..n {
        r[i] = b[i] - ap[i];
    }
    let mut p = r.clone();
    let mut rsold = dot(&r, &r);
    let bnorm2 = dot(b, b).max(1e-30);
    let iters = fixed_iters.unwrap_or(max_iter);
    for _ in 0..iters {
        if fixed_iters.is_none() && rsold <= tol * tol * bnorm2 {
            break;
        }
        a.matvec(&p, &mut ap);
        let denom = dot(&p, &ap);
        if !denom.is_finite() || libm_ops::fabs(denom) < 1e-300 {
            return false;
        }
        let alpha = rsold / denom;
        axpy(alpha, &p, x);
        axpy(-alpha, &ap, &mut r);
        let rsnew = dot(&r, &r);
        if !rsnew.is_finite() {
            return false;
        }
        let beta = rsnew / rsold;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }
    x.iter().all(|v| v.is_finite())
}

/// Thomas algorithm for a tridiagonal system.
///
/// `lower[i]` is a_{i,i-1} for i≥1 (lower[0] unused), `diag`, `upper[i]` is
/// a_{i,i+1} (upper[n-1] unused). Solves in place into `rhs` → solution.
pub fn thomas_solve(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &mut [f64]) -> bool {
    let n = diag.len();
    if n == 0 {
        return true;
    }
    assert_eq!(lower.len(), n);
    assert_eq!(upper.len(), n);
    assert_eq!(rhs.len(), n);
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    if libm_ops::fabs(diag[0]) < 1e-300 {
        return false;
    }
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];
    for i in 1..n {
        let denom = diag[i] - lower[i] * c_prime[i - 1];
        if libm_ops::fabs(denom) < 1e-300 {
            return false;
        }
        c_prime[i] = if i + 1 < n { upper[i] / denom } else { 0.0 };
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom;
    }
    rhs[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        rhs[i] = d_prime[i] - c_prime[i] * rhs[i + 1];
    }
    rhs.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_solves_diagonal() {
        let a = CsrMatrix::from_coo(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 4.0]);
        let b = [2.0, 6.0, 12.0];
        let mut x = [0.0; 3];
        assert!(cg_solve(&a, &b, &mut x, 1e-14, 10, Some(3)));
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn thomas_poisson_1d() {
        // -u'' = 1 on interior, u(0)=u(1)=0 → u = x(1-x)/2
        let n = 5;
        let h = 1.0 / (n as f64 + 1.0);
        let h2 = h * h;
        let mut lower = vec![0.0; n];
        let mut diag = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            lower[i] = if i > 0 { -1.0 / h2 } else { 0.0 };
            upper[i] = if i + 1 < n { -1.0 / h2 } else { 0.0 };
            diag[i] = 2.0 / h2;
            rhs[i] = 1.0;
        }
        assert!(thomas_solve(&lower, &diag, &upper, &mut rhs));
        for i in 0..n {
            let x = (i as f64 + 1.0) * h;
            let expect = 0.5 * x * (1.0 - x);
            assert!((rhs[i] - expect).abs() < 1e-10, "{} vs {}", rhs[i], expect);
        }
    }
}
