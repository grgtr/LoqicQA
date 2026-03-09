"""Global configuration for LogicQA, loaded from config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml

@dataclass
class TestingConfig:
    mode: str = "all"                          # "all" | "specific" | "random"
    specific_samples: List[str] = field(default_factory=list)
    random_count: int = 10
    random_seed: int = 42
    interleave: bool = True

@dataclass
class InternVLConfig:
    model_name: str = "OpenGVLab/InternVL2_5-38B"
    temperature: float = 0.2
    top_p: float = 0.7
    repetition_penalty: float = 1.1
    do_sample: bool = True
    max_new_tokens: int = 512
    device_map: str = "auto"


@dataclass
class GPT4oConfig:
    model_name: str = "gpt-4o"
    temperature: float = 1.0


@dataclass
class GeminiConfig:
    model_name: str = "gemini-1.5-flash"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40


@dataclass
class VLMConfig:
    backend: str = "internvl"
    internvl: InternVLConfig = field(default_factory=InternVLConfig)
    gpt4o: GPT4oConfig = field(default_factory=GPT4oConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


@dataclass
class PipelineConfig:
    n_shots: int = 5
    n_questions: int = 5
    n_sub_questions: int = 5
    question_filter_threshold: float = 0.8


@dataclass
class BPMConfig:
    enabled: bool = True
    patch_size: int = 16
    threshold: float = 0.05


@dataclass
class LangSAMConfig:
    enabled: bool = True
    prompts: Dict[str, str] = field(default_factory=lambda: {
        "pushpins": "The individual black compartments within the transparent plastic storage box",
        "splicing_connectors": "Connector Block",
    })


@dataclass
class PreprocessingConfig:
    bpm: BPMConfig = field(default_factory=BPMConfig)
    langsam: LangSAMConfig = field(default_factory=LangSAMConfig)


@dataclass
class DatasetConfig:
    name: str = "mvtec_loco"
    data_dir: str = "~/dataset-ninja/"
    download_if_missing: bool = True


@dataclass
class OutputConfig:
    results_dir: str = "results"
    save_questions: bool = True
    save_per_image: bool = True


@dataclass
class LogicQAConfig:
    testing: TestingConfig = field(default_factory=TestingConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "LogicQAConfig":
        """Load config from a YAML file, merging with defaults."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = cls()

        if "vlm" in raw:
            v = raw["vlm"]
            cfg.vlm.backend = v.get("backend", cfg.vlm.backend)
            if "internvl" in v:
                for k, val in v["internvl"].items():
                    setattr(cfg.vlm.internvl, k, val)
            if "gpt4o" in v:
                for k, val in v["gpt4o"].items():
                    setattr(cfg.vlm.gpt4o, k, val)
            if "gemini" in v:
                for k, val in v["gemini"].items():
                    setattr(cfg.vlm.gemini, k, val)
        if "pipeline" in raw:
            print("[DEBUG] pipeline", raw["pipeline"])
            for k, val in raw["pipeline"].items():
                setattr(cfg.pipeline, k, val)

        if "dataset" in raw:
            for k, val in raw["dataset"].items():
                setattr(cfg.dataset, k, val)

        if "preprocessing" in raw:
            pp = raw["preprocessing"]
            if "bpm" in pp:
                for k, val in pp["bpm"].items():
                    setattr(cfg.preprocessing.bpm, k, val)
            if "langsam" in pp:
                ls = pp["langsam"]
                cfg.preprocessing.langsam.enabled = ls.get("enabled", True)
                if "prompts" in ls:
                    cfg.preprocessing.langsam.prompts.update(ls["prompts"])

        if "output" in raw:
            for k, val in raw["output"].items():
                setattr(cfg.output, k, val)
        
        if "testing" in raw:
            t = raw["testing"]
            cfg.testing = TestingConfig(
                mode=t.get("mode", "all"),
                specific_samples=t.get("specific_samples", []),
                random_count=t.get("random_count", 10),
                random_seed=t.get("random_seed", 42),
                interleave=t.get("interleave", True),
            )

        return cfg

    @classmethod
    def default(cls) -> "LogicQAConfig":
        return cls()
