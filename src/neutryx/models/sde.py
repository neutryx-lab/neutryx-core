from dataclasses import dataclass

@dataclass
class SDE:
    """Base marker for SDEs; implement drift() and diffusion()."""
    def drift(self, t, x): raise NotImplementedError
    def diffusion(self, t, x): raise NotImplementedError
