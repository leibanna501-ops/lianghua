import inspect

def call_legacy(func, /, *args, **kwargs):
    """Call a legacy function but adapt to its signature safely.

    - If the function declares `debug_log` and you didn't pass it, we add [] by default.
    - If the function doesn't accept some kwargs, we drop them (unless it has **kwargs).
    - Keeps positional args as-is.
    """
    sig = inspect.signature(func)
    params = sig.parameters
    # auto-add debug_log if present and not provided
    if "debug_log" in params and "debug_log" not in kwargs:
        kwargs["debug_log"] = []
    # drop unexpected kwargs if function has no **kwargs
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return func(*args, **kwargs)
