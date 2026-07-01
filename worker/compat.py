"""Compatibility shims for Kaggle / mixed PyTorch stacks."""

from __future__ import annotations

import sys


def ensure_torchvision_functional_tensor() -> None:
    """
    basicsr (GFPGAN / Real-ESRGAN) imports torchvision.transforms.functional_tensor,
    removed in torchvision 0.17+. Map it to functional on newer builds.
    """
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional


def verify_ml_dependency_stack() -> None:
    """Fail fast with a clear message if peft/transformers versions are incompatible."""
    import peft
    import transformers

    peft_ver = tuple(int(x) for x in peft.__version__.split(".")[:2])
    if peft_ver >= (0, 13):
        raise RuntimeError(
            f"peft {peft.__version__} requires transformers with EncoderDecoderCache. "
            "Install compatible pins: pip install 'peft==0.11.1' 'transformers==4.40.2'"
        )

    try:
        from transformers import EncoderDecoderCache  # noqa: F401
    except ImportError:
        # peft < 0.13 does not need EncoderDecoderCache — this is fine.
        pass

    print(
        f"ML deps OK: transformers={transformers.__version__}, peft={peft.__version__}"
    )
