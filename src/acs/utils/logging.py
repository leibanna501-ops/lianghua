import logging, time, json, os
from functools import wraps

def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s"
    )
    return logging.getLogger("acs")

def timeit(msg: str):
    def deco(fn):
        @wraps(fn)
        def wrap(*a, **kw):
            t0 = time.time()
            try:
                return fn(*a, **kw)
            finally:
                dt = time.time() - t0
                logging.getLogger("acs").info(json.dumps({"event":"timing","stage":msg,"sec":round(dt,3)}))
        return wrap
    return deco
