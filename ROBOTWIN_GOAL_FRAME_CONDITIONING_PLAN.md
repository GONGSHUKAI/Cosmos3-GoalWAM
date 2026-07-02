# RoboTwin Goal Frame Conditioning Plan

This is a planning document only. It does not propose rushing into code changes.

## Objective

Add goal-frame conditioning for Cosmos3-Nano-RoboTwin policy training and evaluation.

The policy condition mode must be explicit and consistent between training and evaluation:

- `cond="text_cond"`: text only. This is the current baseline.
- `cond="goal_frame_cond"`: goal image only. No task/subtask text should be used as a policy condition.
- `cond="text_goal_frame_cond"`: text plus goal image.

The goal image layout must also be explicit and consistent between training and evaluation:

- `goal_layout="concat"`: goal image is the same three-camera concat view used by the current RoboTwin policy.
- `goal_layout="cam_high"`: goal image is only the high/head camera.

Both values should be saved in training config/checkpoint metadata and auto-resolved by evaluation whenever possible. Evaluation should fail loudly if the requested runtime condition does not match the trained policy condition.

For this feature branch, assume `action_normalization="meanstd"` and `downsample_video_frames=true` by default and do not maintain separate non-normalized or non-downsampled variants. The canonical Robotwin goal-frame scripts should therefore be named by the condition only, for example `scripts/train/train_robotwin_text_goal_frame_cond.sh`, without repeating the action-normalization/downsample choices in the filename.

## Current Code Facts

The current RoboTwin training dataset is `RoboTwinLeRobotDataset`; it builds each training sample from a local episode window in `cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py:287`, chooses text from task captions in `cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py:299` and `cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py:308`, and emits the final sample dictionary at `cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py:317`.

The current three-camera concat layout is implemented in training at `cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py:366` and mirrored in the evaluation server at `cosmos_framework/scripts/action_policy_server_robotwin.py:415`.

The current action SFT transform tokenizes text and builds a `SequencePlan` in `cosmos_framework/data/vfm/action/transforms.py:661` and `cosmos_framework/data/vfm/action/transforms.py:680`. Current policy mode conditions only on the first VAE vision latent frame in `cosmos_framework/data/vfm/action/transforms.py:302`.

The current action policy recipe uses the Nano model config, action dataset factory, and Robotwin dataset overrides in `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robotwin_nano.py:26`, `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robotwin_nano.py:223`, and `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robotwin_nano.py:188`.

The current TOML schema forbids unknown keys via pydantic `extra="forbid"` in `cosmos_framework/configs/toml_config/sft_config.py:24`, so new TOML-facing knobs need schema support or must be passed only through Hydra tail overrides.

The current evaluation server request protocol includes prompt/current cameras/state, not goal cameras, in `cosmos_framework/scripts/action_policy_server_robotwin.py:25`, and its sample builder creates the policy input in `cosmos_framework/scripts/action_policy_server_robotwin.py:429`.

The local RoboTwin Cosmos client currently sends only current observation fields in `RoboTwin/policy/cosmos_policy/deploy_policy.py:123`. The local RoboTwin eval script runs the expert check but does not preserve the expert final observation in `RoboTwin/script/eval_policy.py:222`.

GoalWAM already demonstrates the important evaluation pattern: capture the successful expert final observation at `/pfs/pfs-7jnepv/shukaigong/code/GoalWAM/third_party/RoboTwin/script/eval_policy.py:334`, then provide it to the policy after reset at `/pfs/pfs-7jnepv/shukaigong/code/GoalWAM/third_party/RoboTwin/script/eval_policy.py:459`.

## Reasoner Branch Assessment

Using the Qwen3-VL/reasoner visual pathway for the goal frame is conceptually the right direction. It resembles text conditioning more than appending a goal frame to the 9-frame or 33-frame VAE video sequence, and it avoids disturbing the VAE temporal shape.

This is also consistent with the Cosmos3 report. In `Cosmos3.pdf`, Sec. 2.1.1 describes two visual encoders: a ViT-style encoder for visual understanding and a frozen Wan2.2 VAE for visual generation. Sec. 2.2.1 states that the AR subsequence carries language plus ViT-encoded image/video tokens, while the diffusion subsequence carries VAE image/video tokens plus audio/action tokens. Sec. 2.3.2 states that diffusion/generator tokens attend to AR/reasoner context, while AR tokens do not attend back to diffusion tokens. For goal-frame policy conditioning, that implies the goal image belongs in the AR/reasoner prefix, not in the VAE video trajectory.

However, it is not already wired into the action-policy training path as a simple switch.

The standalone reasoner path can accept images: `generate_reasoner_text` takes `pixel_values` and `image_grid_thw` in `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py:275`, and the high-level multimodal prompt path builds image-plus-text messages in `cosmos_framework/model/vfm/omni_mot_model.py:4052`.

The action-policy path is different. The packed generation network currently embeds text IDs with `embed_tokens` in `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py:585`, VAE latents with `vae2llm` in `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py:606`, and actions with `action2llm` in `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py:740`. It does not currently scatter Qwen visual embeddings into the packed action/video generation sequence.

The Nano model config does have a Qwen3-VL model identity in `cosmos_framework/configs/base/experiment/sft/models/nano_model_config.py:113`, and `Qwen3VLTextForCausalLM` can materialize a visual tower when its config includes one in `cosmos_framework/model/vfm/mot/unified_mot.py:1968`. But the current Nano tokenizer config uses an `LLMTokenizerProcessor` in `cosmos_framework/configs/base/experiment/sft/models/nano_model_config.py:141`, while the full multimodal processor factory is `build_processor` in `cosmos_framework/data/vfm/processors/__init__.py:100`.

Conclusion: yes, route goal images through the reasoner-style visual branch, but implement it as an AR-prefix conditioning extension inside the packed generator path. Do not rely on the standalone `generate_reasoner_text` API, because that API bypasses VAE/action heads and only generates text.

## Recommended Architecture

Add a new AR/reasoner "condition prefix" for goal images that is packed before the VAE video/action diffusion tokens, analogous to text tokens.

For `text_cond`, keep the current path:

- tokenize `ai_caption`;
- no goal image fields;
- no Qwen visual image tokens.

For `goal_frame_cond`:

- load the episode-final goal image;
- process it through a Qwen3-VL image processor;
- include image placeholder tokens plus Qwen ViT visual embeddings in the AR/reasoner prefix;
- do not include task/subtask text tokens as a condition.

For `text_goal_frame_cond`:

- load the episode-final goal image;
- build a multimodal user message containing image plus task/subtask text;
- process it through Qwen3-VL chat/template logic;
- include both image placeholder tokens and text tokens in the AR/reasoner prefix.

The VAE video/action part should remain the current policy diffusion sequence. Do not append a goal image to the VAE video frames. This keeps `encode_exact_durations=[33]` or `[9]` unchanged, preserving the temporal compression assumptions set in `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robotwin_nano.py:249`.

## Implementation Plan

1. Add shared enums/config names.

Use exact string enums:

- `cond`: `text_cond`, `goal_frame_cond`, `text_goal_frame_cond`.
- `goal_layout`: `concat`, `cam_high`.

Add validation close to the dataset/factory path first, then expose through TOML/schema if we want first-class recipe keys. Because the TOML schema forbids unknown keys at `cosmos_framework/configs/toml_config/sft_config.py:24`, either add schema fields or pass these via existing Hydra tail overrides until the schema is extended.

Default dataset/training overrides for all new goal-frame experiments:

- `action_normalization="meanstd"`;
- `downsample_video_frames=true`;
- `model.config.tokenizer.encode_exact_durations=[9]`.

Do not plan or test extra goal-frame variants for raw actions or full 33-frame video unless a later ablation explicitly asks for them.

2. Add goal-frame loading to `RoboTwinLeRobotDataset`.

Extend `RoboTwinLeRobotDataset.__init__` in `cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py:84` with:

- `cond: str = "text_cond"`;
- `goal_layout: str = "concat"`;
- optional `goal_frame_source: str = "episode_last"`.

When `cond` uses a goal image, decode the actual final frame of the episode, not the last frame of the sampled training window. This matches the GoalWAM fix at `/pfs/pfs-7jnepv/shukaigong/code/GoalWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py:311`.

For `goal_layout="concat"`, reuse the same three-camera geometry as `cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py:366`. For `goal_layout="cam_high"`, load only the head/high camera using the same camera key family already defined in `cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py:52`.

Add a `goal_image` field to the raw sample. Keep current `video` unchanged.

3. Add condition-mode handling in `ActionTransformPipeline`.

Extend `ActionTransformPipeline` so it receives `cond` and `goal_layout`.

For `text_cond`, keep current behavior.

For `goal_frame_cond`, suppress text conditioning deliberately. Do not pass an empty task string through the normal prompt template, because that still creates structural text prompt tokens. Use an explicit no-text path.

For `text_goal_frame_cond`, keep the existing `ai_caption` text and add goal-image tokens/features.

The current `build_sequence_plan_from_mode` takes `has_text=True` by default in `cosmos_framework/data/vfm/action/transforms.py:235`; update the caller so `has_text` reflects the selected `cond`. If the image-prefix path needs a new flag, introduce something like `has_goal_image_context` on `SequencePlan` rather than overloading `has_vision`, because `has_vision` currently means VAE generation vision tokens in `cosmos_framework/data/vfm/sequence_packing/types.py:320`.

4. Add Qwen image-prefix support to packing/model input.

This is the main model/dataflow change.

The implementation should model the goal frame as AR/reasoner context, not as an additional clean DM/VAE conditioning frame. The current `SequencePlan.condition_frame_indexes_vision` mechanism still applies to the policy's observed current-frame VAE condition, but should not be reused to represent the goal frame.

Add fields to the sample/batch contract for Qwen goal-image conditioning, likely:

- `goal_input_ids`;
- `goal_attention_mask`;
- `goal_pixel_values`;
- `goal_image_grid_thw`;
- optionally a unified `condition_input_ids` if text and image are processed together.

The Qwen3-VL processor already returns `input_ids`, `attention_mask`, `pixel_values`, and `image_grid_thw` for image prompts in `cosmos_framework/data/vfm/processors/qwen3vl_processor.py:43`.

Then extend sequence packing to reserve AR prefix positions for this multimodal condition. The current `pack_text_tokens` only knows text token IDs in `cosmos_framework/data/vfm/sequence_packing/modalities.py:146`; it will need either:

- a new `pack_condition_tokens` that can accept precomputed embeddings for image-placeholder positions; or
- an extension of `pack_text_tokens` plus a model-side scatter step that replaces image placeholder embeddings with Qwen visual embeddings before the transformer.

The second option is closer to Qwen3-VL's native flow, where multimodal inputs scatter visual embeddings into placeholder token positions.

5. Extend `Cosmos3VFMNetwork._encode_text` or add `_encode_condition_prefix`.

Current `_encode_text` embeds token IDs only through `self.language_model.model.embed_tokens` in `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py:597`.

Add a condition-prefix encoder that:

- embeds text and image placeholder token IDs;
- when goal image fields are present, runs the Qwen visual tower to obtain image embeddings;
- scatters image embeddings into the placeholder token positions;
- writes the resulting prefix embeddings into `packed_sequence`.

The visual tower exists on `Qwen3VLTextForCausalLM` when configured in `cosmos_framework/model/vfm/mot/unified_mot.py:1971`. If the loaded checkpoint/config lacks `visual`, raise a clear error for `goal_frame_cond` and `text_goal_frame_cond`.

6. Make trainable-parameter policy explicit.

Freeze the Qwen/ViT visual tower and train the existing generation/action path. This is now the default design, not just an option. It keeps the experiment close to current Cosmos3-Nano-RoboTwin policy training while allowing the generator/action path to learn to use the AR goal-image context.

Update optimizer `keys_to_select` in `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robotwin_nano.py:73` only if new trainable modules are added.

7. Extend evaluation server protocol.

Add server arguments:

- `--cond {text_cond,goal_frame_cond,text_goal_frame_cond}`;
- `--goal-layout {concat,cam_high}`;
- `--goal-source {oracle,generated}`;
- `--cond auto` and `--goal-layout auto` are acceptable if they resolve from the saved training config.

The server already has config-reading helpers such as `_read_bool_from_dataloader_config` in `cosmos_framework/scripts/action_policy_server_robotwin.py:188`; add string readers for `cond` and `goal_layout`.

Extend request payload fields:

- for `goal_layout="concat"`: accept `goal_head`, `goal_left`, `goal_right`;
- for `goal_layout="cam_high"`: accept `goal_head` only.

For `text_cond`, reject unexpected goal fields or ignore with a warning. For goal modes, require matching goal fields. Default `goal_source` to `oracle` for current RoboTwin verification, but keep the request/server structure open for a future `generated` goal image supplied by a planner or subgoal-image generator.

8. Patch RoboTwin client/eval to provide oracle goal observations.

Patch `RoboTwin/policy/cosmos_policy/deploy_policy.py` so the client caches an episode goal observation and sends goal camera arrays with every policy request, similar to GoalWAM's `set_episode_ref_obs` pattern.

Patch `RoboTwin/script/eval_policy.py` so after a successful rule-based expert execution it captures the final observation before closing/resetting the env. This follows GoalWAM's pattern at `/pfs/pfs-7jnepv/shukaigong/code/GoalWAM/third_party/RoboTwin/script/eval_policy.py:334` and `/pfs/pfs-7jnepv/shukaigong/code/GoalWAM/third_party/RoboTwin/script/eval_policy.py:459`.

Default evaluation uses the oracle expert-final goal image. Leave a clear extension point for generated goal images: the client/server should be able to accept an externally supplied generated goal image with the same `goal_layout` contract, even if the first implementation only populates it from the rule-based expert final observation.

9. Add train/eval scripts and recipes.

Add new script variants instead of modifying the baseline scripts in place:

- `scripts/train/train_robotwin_goal_frame_cond.sh`;
- `scripts/train/train_robotwin_text_goal_frame_cond.sh`;
- `scripts/eval/eval_robotwin_goal_frame_cond.sh`;
- `scripts/eval/eval_robotwin_text_goal_frame_cond.sh`.

These scripts should assume action normalization and video downsampling are enabled. The current downsample/action-normalized train script sets the relevant dataset overrides in `scripts/train/train_robotwin_downsample_action_norm.sh:31`; use those settings as the baseline and add only `cond`, `goal_layout`, and goal-source-related overrides in the new scripts.

The current TOML recipe is `examples/toml/sft_config/action_policy_robotwin.toml:18`. Either add new TOMLs or keep one TOML and select condition/layout via shell overrides.

10. Add verification tests.

Add unit or smoke tests for:

- dataset loads actual episode-final goal image, not sampled-window last image;
- `goal_layout="concat"` shape matches the current concat training/eval layout;
- `goal_layout="cam_high"` shape matches processor expectations;
- `cond="text_cond"` produces no goal-image fields;
- `cond="goal_frame_cond"` produces no task/subtask text condition;
- `cond="text_goal_frame_cond"` produces both text and goal-image condition;
- server rejects mismatched training/eval `cond` and `goal_layout`;
- RoboTwin eval captures and sends the expert final observation.

## Important Differences From GoalWAM

GoalWAM's main `image_b_conditioning=True` path replaces text with image-B conditioning. That is not the same as `text_goal_frame_cond`. GoalWAM's joint text-plus-image mode is closer to this plan and is configured at `/pfs/pfs-7jnepv/shukaigong/code/GoalWAM/configs/model/fastwam_joint_text_image_b.yaml:11`.

GoalWAM has a cam-high-only route. For Cosmos, `goal_layout="concat"` should be the default because the current policy visual observation is the three-camera concat.

Robotwin verification uses an oracle expert final frame. This is useful for checking whether goal conditioning helps, but it is not a full long-horizon deployment test until the goal image comes from a planner or subgoal-image generator.

## Open Design Questions

1. Should `goal_frame_cond` include any generic structural text such as "Given the goal image, predict the action"? Recommended: no task/subtask text, but a minimal fixed system/user scaffold may be unavoidable for Qwen's chat template. Keep it identical across all samples and document it.

2. What `max_pixels` should goal images use? Qwen3-VL image processing can produce thousands of image tokens; the example processor output shows 2752 image placeholder tokens for one image in `cosmos_framework/data/vfm/processors/qwen3vl_processor.py:159`. We should cap pixels for both `concat` and `cam_high` to keep action sequence length and memory stable.

3. Should eval auto-infer `cond` and `goal_layout` from the checkpoint config, or require explicit CLI args? Recommended: support auto, but fail if explicit args conflict with checkpoint config.
