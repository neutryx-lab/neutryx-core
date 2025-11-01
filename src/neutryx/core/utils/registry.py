REGISTRY = {}

def register(name=None):
    def deco(obj):
        key = name or obj.__name__
        if key in REGISTRY:
            raise ValueError(f"Duplicate registry key: {key}")
        REGISTRY[key] = obj
        return obj
    return deco

def get(name):
    return REGISTRY[name]

def available():
    return sorted(REGISTRY.keys())