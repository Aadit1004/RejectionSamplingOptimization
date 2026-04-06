import time

def time_function(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed = end - start
    return result, elapsed


# dont really need this but there if want to use
def summarize_sampling(acceptance_count, n_samples, elapsed_time):
    acceptance_rate = acceptance_count / n_samples
    return {
        "accepted": acceptance_count,
        "total": n_samples,
        "acceptance_rate": acceptance_rate,
        "elapsed_time_sec": elapsed_time,
    }