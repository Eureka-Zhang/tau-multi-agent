# Copyright Sierra

import os
import argparse
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env",
        type=str,
        choices=["retail", "airline"],
        default=os.environ.get("TAU_ENV", "retail"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("TAU_MODEL"),
        help="The model to use for the agent (default: $TAU_MODEL)",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        default=os.environ.get("TAU_MODEL_PROVIDER"),
        help="The model provider for the agent (default: $TAU_MODEL_PROVIDER)",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default=os.environ.get("TAU_USER_MODEL", "gpt-4o"),
        help="The model to use for the user simulator (default: $TAU_USER_MODEL or gpt-4o)",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        default=os.environ.get("TAU_USER_MODEL_PROVIDER"),
        help="The model provider for the user simulator (default: $TAU_USER_MODEL_PROVIDER)",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot", "two-agent-handoff"],
    )
    parser.add_argument(
        "--handoff-method",
        type=str,
        default="structured",
        choices=["structured"],
        help="Handoff package method for the two-agent-handoff strategy",
    )
    # Per-role model overrides for two-agent-handoff. When unset they fall back
    # to --model / --model-provider.
    parser.add_argument(
        "--agent-a-model",
        type=str,
        default=os.environ.get("TAU_AGENT_A_MODEL"),
        help="Model for Agent A (info agent). Falls back to --model. ($TAU_AGENT_A_MODEL)",
    )
    parser.add_argument(
        "--agent-a-model-provider",
        type=str,
        choices=provider_list,
        default=os.environ.get("TAU_AGENT_A_MODEL_PROVIDER"),
        help="Provider for Agent A. Falls back to --model-provider. ($TAU_AGENT_A_MODEL_PROVIDER)",
    )
    parser.add_argument(
        "--agent-b-model",
        type=str,
        default=os.environ.get("TAU_AGENT_B_MODEL"),
        help="Model for Agent B (execution agent). Falls back to --model. ($TAU_AGENT_B_MODEL)",
    )
    parser.add_argument(
        "--agent-b-model-provider",
        type=str,
        choices=provider_list,
        default=os.environ.get("TAU_AGENT_B_MODEL_PROVIDER"),
        help="Provider for Agent B. Falls back to --model-provider. ($TAU_AGENT_B_MODEL_PROVIDER)",
    )
    parser.add_argument(
        "--handoff-builder-model",
        type=str,
        default=os.environ.get("TAU_HANDOFF_BUILDER_MODEL"),
        help="Model for the handoff package compressor. Falls back to --model. ($TAU_HANDOFF_BUILDER_MODEL)",
    )
    parser.add_argument(
        "--handoff-builder-model-provider",
        type=str,
        choices=provider_list,
        default=os.environ.get("TAU_HANDOFF_BUILDER_MODEL_PROVIDER"),
        help="Provider for the handoff compressor. Falls back to --model-provider. ($TAU_HANDOFF_BUILDER_MODEL_PROVIDER)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    args = parser.parse_args()

    if args.agent_strategy == "two-agent-handoff":
        # Each role can be configured entirely via .env. A role-specific value
        # takes precedence; otherwise it falls back to the base TAU_MODEL.
        a_model = args.agent_a_model or args.model
        a_provider = args.agent_a_model_provider or args.model_provider
        b_model = args.agent_b_model or args.model
        b_provider = args.agent_b_model_provider or args.model_provider
        builder_model = args.handoff_builder_model or args.model
        builder_provider = args.handoff_builder_model_provider or args.model_provider

        missing = []
        if not a_model:
            missing.append("Agent A model (TAU_AGENT_A_MODEL or TAU_MODEL / --agent-a-model)")
        if not a_provider:
            missing.append("Agent A provider (TAU_AGENT_A_MODEL_PROVIDER or TAU_MODEL_PROVIDER)")
        if not b_model:
            missing.append("Agent B model (TAU_AGENT_B_MODEL or TAU_MODEL / --agent-b-model)")
        if not b_provider:
            missing.append("Agent B provider (TAU_AGENT_B_MODEL_PROVIDER or TAU_MODEL_PROVIDER)")
        if not builder_model:
            missing.append("Handoff builder model (TAU_HANDOFF_BUILDER_MODEL or TAU_MODEL)")
        if not builder_provider:
            missing.append("Handoff builder provider (TAU_HANDOFF_BUILDER_MODEL_PROVIDER or TAU_MODEL_PROVIDER)")
        if not args.user_model:
            missing.append("User model (TAU_USER_MODEL or --user-model)")
        if not args.user_model_provider:
            missing.append("User provider (TAU_USER_MODEL_PROVIDER or --user-model-provider)")
        if missing:
            parser.error(
                "Missing required model configuration for two-agent-handoff:\n  - "
                + "\n  - ".join(missing)
            )

        # Ensure the base model/provider are populated (used for checkpoint
        # naming and as the resolved fallback downstream) even when only the
        # per-role variables were provided in .env.
        args.model = args.model or a_model
        args.model_provider = args.model_provider or a_provider
    else:
        missing = []
        if not args.model:
            missing.append("--model (or TAU_MODEL in .env)")
        if not args.model_provider:
            missing.append("--model-provider (or TAU_MODEL_PROVIDER in .env)")
        if not args.user_model_provider:
            missing.append("--user-model-provider (or TAU_USER_MODEL_PROVIDER in .env)")
        if missing:
            parser.error("Missing required configuration: " + ", ".join(missing))

    print(args)
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        handoff_method=args.handoff_method,
        agent_a_model=args.agent_a_model,
        agent_a_model_provider=args.agent_a_model_provider,
        agent_b_model=args.agent_b_model,
        agent_b_model_provider=args.agent_b_model_provider,
        handoff_builder_model=args.handoff_builder_model,
        handoff_builder_model_provider=args.handoff_builder_model_provider,
    )


def main():
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
