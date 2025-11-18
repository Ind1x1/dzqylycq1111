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

