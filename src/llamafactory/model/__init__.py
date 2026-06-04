# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "QuantizationMethod",
    "load_config",
    "load_model",
    "load_tokenizer",
    "find_all_linear_modules",
    "load_valuehead_params",
]


def __getattr__(name: str):
    if name in {"load_config", "load_model", "load_tokenizer"}:
        from .loader import load_config, load_model, load_tokenizer

        return {"load_config": load_config, "load_model": load_model, "load_tokenizer": load_tokenizer}[name]

    if name == "find_all_linear_modules":
        from .model_utils.misc import find_all_linear_modules

        return find_all_linear_modules

    if name == "QuantizationMethod":
        from .model_utils.quantization import QuantizationMethod

        return QuantizationMethod

    if name == "load_valuehead_params":
        from .model_utils.valuehead import load_valuehead_params

        return load_valuehead_params

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
