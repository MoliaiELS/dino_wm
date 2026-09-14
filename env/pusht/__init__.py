"""PushT package with lazy simulator imports."""

__all__ = ["PushTEnv"]


def __getattr__(name):
    if name == "PushTEnv":
        from .pusht_env import PushTEnv

        return PushTEnv
    raise AttributeError(name)
