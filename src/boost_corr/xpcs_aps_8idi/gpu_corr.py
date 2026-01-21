from .gpu_corr_multitau import solve_multitau
from .gpu_corr_twotime import solve_twotime
from .xpcs_result import XpcsResult


def solve_corr(analysis='Both', *args, **kwargs):
    assert analysis in ['Multitau', 'Twotime', 'Both']
    if analysis == 'Multitau':
        return solve_multitau(*args, save_results=True, **kwargs)
    elif analysis == 'Twotime':
        return solve_twotime(*args, save_results=True, **kwargs)
    elif analysis == 'Both':
        rf_kwargs_m, payload_m = solve_multitau(*args, save_results=False, **kwargs)
        rf_kwargs_t, payload_t = solve_twotime(*args, save_results=False, **kwargs)
        rf_kwargs_m.update(rf_kwargs_t)
        with XpcsResult(**rf_kwargs_m) as results:
            for item in payload_m:
                results.append(item)
            for item in payload_t[1]:   # skip norm_scattering
                results.append(item)
        
        if results.success:
            return results.fname
        else:
            return None

