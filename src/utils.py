from time import perf_counter
import functools


COLORS = {
    5: "\033[95m",  
    7: "\033[94m",
    9: "\033[93m",
    11: "\033[92m",
    13: "\033[91m",  
}

RESET = "\033[0m"

def get_color(run_time):

    if run_time >= 13:
        return COLORS[13]
    elif run_time >= 11:
        return COLORS[11]
    elif run_time >= 9:
        return COLORS[9]
    elif run_time >= 7:
        return COLORS[7]
    elif run_time >= 5:
        return COLORS[5]
    else:
        return ""  

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = perf_counter()
        value = func(*args, **kwargs)
        end_time = perf_counter()
        run_time = end_time - start_time
        color = get_color(run_time)
        print(f"{color}[TIMER] Finished {func.__name__}() in {run_time:.4f} secs{RESET}")
        return value
    return wrapper_timer



