#!python
import os
import sys
sys.path.append(os.getcwd())
import emcee
import numpy as np
from subprocess import Popen


from setup import best_filename, bk_filename, fit_config, out_filename
from astropy.table import Table

filename = bk_filename
Popen(["cp", filename, "temp.h5"]).wait()
filename = "temp.h5"

reader = emcee.backends.HDFBackend(filename)
chain = reader.get_chain(flat=True)
log_prob = reader.get_log_prob(flat=True)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

print(min(log_prob / -0.5 * fit_config.temperature))
print(chain[-5:])

tau = reader.get_autocorr_time(quiet=True)
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))


sampler = reader
_chain = sampler.get_chain(flat=True)
_log_prob = sampler.get_log_prob(flat=True)
_blobs = sampler.get_blobs(flat=True)
output = np.vstack(
    (
        _log_prob / -0.5 * fit_config.temperature,
        _chain.T,
        *[_blobs[i] for i in fit_config.blobs],
    )
).T
header = ["chi2"] + fit_config.parameters_to_fit + fit_config.blobs
print(header)
print(output[0])
tab = Table(
    data=output,
    names=header,
    dtype=len(header) * ["f4"],
)
for x in fit_config.param_fix:
    tab[x] = fit_config.param_fix[x]


id = np.argmin(tab["chi2"])
print(tab[id:id+1].to_pandas().to_dict("records")[0])
tab[id:id+1].write(best_filename, overwrite=True)

formatter = {i: ".7f" for i in header}
for i in formatter.keys():
    tab[i].info.format = formatter[i]
tab.write(out_filename, format="csv", overwrite=True)

