import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import stats as S
import numpy as np
from scipy import stats
data = S.main
X1, y, n1 = S.design(data, True)
b, rss, df = S.ols(X1, y)
s2 = rss / df
cov = s2 * np.linalg.inv(X1.T @ X1)
se = np.sqrt(np.diag(cov)); t = stats.t.ppf(0.975, df)
print("df", df, "resid sd", s2 ** 0.5)
for n, bb, ss in zip(n1, b, se):
    if "x" in n: print("%-8s %6.2f  [%5.2f, %5.2f]" % (n, bb, bb - t * ss, bb + t * ss))
# minimal detectable interaction (80% power, alpha .05) for single contrast via noncentral F on the 6-df test
from scipy.stats import ncf, f
Fcrit = f.ppf(0.95, 6, df)
for lam in np.arange(2, 40, 0.5):
    if 1 - ncf.cdf(Fcrit, 6, df, lam) >= 0.8:
        print("noncentrality for 80% power:", lam); break
# lambda = SS_int / sigma^2 ; typical interaction coefficient size if all 6 equal magnitude d: SS ~ 6*d^2 * (leverage ~ 0.5? ) -> report RMS
rms_needed = (lam * s2 / 6) ** 0.5
print("approx RMS interaction coefficient detectable at 80%% power (rough): %.2f" % rms_needed)
