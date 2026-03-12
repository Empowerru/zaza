#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kNmToUm = 1.0e-3;   // [um / nm]
constexpr double kLinTol = 1.0e-14;  // regularization floor
constexpr double kExpCap = 600.0;    // avoid overflow in exp()

struct CMat2 {
  std::complex<double> m00{1.0, 0.0};
  std::complex<double> m01{0.0, 0.0};
  std::complex<double> m10{0.0, 0.0};
  std::complex<double> m11{1.0, 0.0};
};

struct CVec2 {
  std::complex<double> v0{0.0, 0.0};
  std::complex<double> v1{0.0, 0.0};
};

CMat2 mul(const CMat2& a, const CMat2& b) {
  CMat2 c;
  c.m00 = a.m00 * b.m00 + a.m01 * b.m10;
  c.m01 = a.m00 * b.m01 + a.m01 * b.m11;
  c.m10 = a.m10 * b.m00 + a.m11 * b.m10;
  c.m11 = a.m10 * b.m01 + a.m11 * b.m11;
  return c;
}

CVec2 mul(const CMat2& a, const CVec2& x) {
  return {a.m00 * x.v0 + a.m01 * x.v1, a.m10 * x.v0 + a.m11 * x.v1};
}

std::complex<double> kappa_branch(std::complex<double> x) {
  std::complex<double> k = std::sqrt(x);
  if (k.imag() < 0.0) {
    k = -k;
  }
  if (std::abs(k) < kLinTol) {
    k = {kLinTol, 0.0};
  }
  return k;
}

}  // namespace

void dima_laser(const std::vector<double>& xl, const std::vector<double>& xr,
                const std::vector<std::complex<double>>& eps, const double lambda, double& R, double& T,
                std::vector<double>& labs) {
  const std::size_t n = eps.size();

  if (xl.size() != n || xr.size() != n) {
    throw std::invalid_argument("dima_laser: xl/xr/eps size mismatch");
  }
  if (lambda <= 0.0) {
    throw std::invalid_argument("dima_laser: lambda must be > 0");
  }

  labs.assign(n, 0.0);
  R = 0.0;
  T = 1.0;
  if (n == 0) {
    return;
  }

  // Normal incidence, s-pol only.
  const double k0 = 2.0 * kPi / lambda;  // [1/um], lambda is [um]

  // Medium indexing:
  //   0      : left semi-infinite vacuum (eps=1)
  //   1..n   : provided cells
  //   n+1    : right semi-infinite medium = eps.back()
  std::vector<std::complex<double>> e(n + 2);
  e[0] = {1.0, 0.0};
  for (std::size_t m = 0; m < n; ++m) {
    e[m + 1] = eps[m];
  }
  e[n + 1] = eps.back();

  // Dimensionless cell lengths dz_bar[j] = k0 * dz_j, j=1..n; dz_bar[0]=0.
  std::vector<double> dz_bar(n + 1, 0.0);
  for (std::size_t m = 0; m < n; ++m) {
    const double d_nm = xr[m] - xl[m];
    if (!(d_nm > 0.0)) {
      throw std::invalid_argument("dima_laser: each cell must satisfy xr[m] > xl[m]");
    }
    dz_bar[m + 1] = k0 * (d_nm * kNmToUm);
  }

  std::vector<std::complex<double>> kappa(n + 2);
  for (std::size_t j = 0; j < n + 2; ++j) {
    kappa[j] = kappa_branch(e[j]);
  }
  if (std::abs(kappa[0]) < kLinTol) {
    kappa[0] = {kLinTol, 0.0};
  }

  std::vector<std::complex<double>> gamma(n + 1);
  std::vector<std::complex<double>> beta(n + 2);
  std::vector<double> sigma(n + 1, 0.0);
  std::vector<CMat2> g_mat(n + 1);

  CMat2 h_hat{};
  double sigma_sum = 0.0;
  std::complex<double> gamma_product{1.0, 0.0};

  for (std::size_t j = 0; j <= n; ++j) {
    const std::complex<double> kj = kappa[j];
    const std::complex<double> kj1 = kappa[j + 1];

    // s-pol gamma_j (Basko Eq. 17 form).
    gamma[j] = kj / kj1;

    // G_j matrix (Basko Eq. 15 form), with dz_bar[0]=0 at the first interface.
    const std::complex<double> delta = std::complex<double>(0.0, 1.0) * kj * dz_bar[j];
    const std::complex<double> exp_p = std::exp(delta);
    const std::complex<double> exp_m = std::exp(-delta);

    CMat2 g{};
    g.m00 = 0.5 * (1.0 + gamma[j]) * exp_p;
    g.m01 = 0.5 * (1.0 - gamma[j]) * exp_m;
    g.m10 = 0.5 * (1.0 - gamma[j]) * exp_p;
    g.m11 = 0.5 * (1.0 + gamma[j]) * exp_m;
    g_mat[j] = g;

    // Appendix-A-like scaling to regularize long/evanescent stacks.
    sigma[j] = std::max(0.0, kappa[j].imag() * dz_bar[j]);
    const double f = std::exp(-sigma[j]);
    CMat2 g_hat{};
    g_hat.m00 = f * g.m00;
    g_hat.m01 = f * g.m01;
    g_hat.m10 = f * g.m10;
    g_hat.m11 = f * g.m11;

    h_hat = mul(g_hat, h_hat);
    sigma_sum += sigma[j];
    gamma_product *= gamma[j];
  }

  if (std::abs(h_hat.m11) < kLinTol) {
    throw std::runtime_error("dima_laser: near-singular transfer matrix");
  }

  // Reflection/transmission amplitudes (Appendix-A form).
  const std::complex<double> r = -h_hat.m10 / h_hat.m11;
  const std::complex<double> p = -std::exp(-sigma_sum) * gamma_product / h_hat.m11;

  // Recover in-layer amplitudes with overflow-safe logarithmic scaling.
  std::vector<CVec2> b(n + 2);
  std::vector<double> scale(n + 2, 0.0);
  b[0] = {{1.0, 0.0}, r};
  for (std::size_t j = 0; j <= n; ++j) {
    const CVec2 raw = mul(g_mat[j], b[j]);
    const double max_a = std::max({std::abs(raw.v0), std::abs(raw.v1), 1.0});
    const double s = std::log(max_a);
    const double inv = std::exp(-s);
    b[j + 1].v0 = raw.v0 * inv;
    b[j + 1].v1 = raw.v1 * inv;
    scale[j + 1] = scale[j] + s;
  }

  // s-pol beta_j (Basko Eq. 28 form).
  for (std::size_t j = 1; j <= n + 1; ++j) {
    beta[j] = kappa[j] / kappa[0];
  }

  R = std::norm(r);
  T = std::max(0.0, beta[n + 1].real() * std::norm(p));

  // Raw per-cell absorption fractions (no Basko rescaling, as requested).
  for (std::size_t j = 1; j <= n; ++j) {
    const double growth = std::exp(std::min(2.0 * scale[j], kExpCap));
    const double two_sigma = 2.0 * sigma[j];

    const double sec_a = std::norm(b[j].v0) * (1.0 - std::exp(-two_sigma));
    const double sec_b = std::norm(b[j].v1) * (std::exp(two_sigma) - 1.0);
    const double f_sec = beta[j].real() * growth * (sec_a + sec_b);

    const std::complex<double> phase =
        std::exp(std::complex<double>(0.0, 2.0 * kappa[j].real() * dz_bar[j])) - std::complex<double>(1.0, 0.0);
    const std::complex<double> inter = b[j].v0 * std::conj(b[j].v1) * phase;
    const double f_osc = 2.0 * beta[j].imag() * growth * inter.imag();

    double q = f_sec + f_osc;
    if (std::abs(q) < 1.0e-15) {
      q = 0.0;
    }
    labs[j - 1] = q;
  }

  // Tiny conservation fix only for roundoff.
  const double q_sum = std::accumulate(labs.begin(), labs.end(), 0.0);
  const double delta = 1.0 - (R + T + q_sum);
  if (std::abs(delta) < 1.0e-10) {
    T += delta;
    if (T < 0.0 && T > -1.0e-12) {
      T = 0.0;
    }
  }
}
