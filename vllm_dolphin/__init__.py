# SPDX-License-Identifier: Apache-2.0

from vllm import ModelRegistry


def register():
    from .dolphin import DolphinForConditionalGeneration

    if "DolphinForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("DolphinForConditionalGeneration", DolphinForConditionalGeneration)
