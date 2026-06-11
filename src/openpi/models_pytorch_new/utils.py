import typing
import torch
import torch.utils.checkpoint

def lru_cache(typed=False):
    import functools
    Tcache = typing.TypeVar("Tcache")
    def decorating_function(user_function: Tcache):
        wrapper = functools._lru_cache_wrapper(user_function, 0, typed, functools._CacheInfo)
        wrapper.cache_parameters = lambda : {'maxsize': 0, 'typed': typed}
        return functools.update_wrapper(wrapper, user_function)
    return typing.cast(Tcache, decorating_function)

def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    if dtype_str == "mp_bfloat16":
        assert False
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "mp_bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_str]

T = typing.TypeVar("T")
@lru_cache()
def checkpointing(fn_module: T, use_reentrant: bool | None = None) -> T:
    def new_func(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn_module, *args, **kwargs, use_reentrant=use_reentrant)
    return typing.cast(T, new_func)
