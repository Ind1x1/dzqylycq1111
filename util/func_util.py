import time
from functools import wraps

from common.log import default_logger as logger

def retry(retry_times=10, retry_interval=5, raise_exception=True):
    """
    Decorator to retry a function on exception.

    Args:
        retry_times (int): Number of times to retry.
        retry_interval (int): Seconds to wait between retries.
        raise_exception (bool): Raise the last exception if all retries fail.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(retry_times):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        f"[retry] Function '{func.__name__}' failed on attempt {attempt+1}/{retry_times} with error: {exc}"
                    )
                    if attempt < retry_times - 1:
                        logger.info(
                            f"[retry] Retrying '{func.__name__}' in {retry_interval} seconds..."
                        )
                        time.sleep(retry_interval)
            logger.error(
                f"[retry] Function '{func.__name__}' failed after {retry_times} attempts."
            )
            if raise_exception and last_exc is not None:
                raise last_exc
            return None
        return wrapper
    return decorator

class TimeoutException(Exception):
    pass


def threading_timeout(secs=-1, callback_func=None):
    """
    Decorator for timeout that limits the execution
    time of functions executed in main and non-main threads
    :param secs: timeout seconds
    :param callback_func: the function that set the timeout
    """
    if callback_func is None:
        if secs <= 0:
            timeout_secs_value = TIMEOUT_MAX
        else:
            timeout_secs_value = secs
    else:
        timeout_secs_value = callback_func()

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout_secs_value)
                except futures.TimeoutError:
                    raise TimeoutException("Function call timed out")
                return result

        return wrapped

    return decorator