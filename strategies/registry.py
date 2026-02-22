class Registry:
    def __init__(self):
        self._storage = {}

    def register(self, name):
        def decorator(cls):
            self._storage[name] = cls
            return cls

        return decorator

    def get(self, name):
        return self._storage.get(name)


MODEL_REGISTRY = Registry()
EVAL_REGISTRY = Registry()
