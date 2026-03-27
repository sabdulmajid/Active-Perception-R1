from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

from active_perception_r1.rollout.zoom_runtime import (
    STATUS_ZOOM_EXECUTED,
    build_observation_message,
    execute_zoom_action,
    malformed_zoom_trace,
    zoom_limit_trace,
)
from active_perception_r1.utils.trace_parser import parse_reasoning_trace

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput


@register("active_perception_zoom_agent")
class ActivePerceptionZoomAgentLoop(ToolAgentLoop):
    """Multi-turn agent loop that treats `<zoom_roi .../>` tags as executable crop actions."""

    def __init__(
        self,
        *args,
        max_zoom_calls: int = 3,
        min_relative_area: float = 0.02,
        max_relative_area: float = 0.65,
        continue_instruction: str = (
            "Continue reasoning. If more visual detail is needed, emit another "
            "<zoom_roi x0=\"...\" y0=\"...\" x1=\"...\" y1=\"...\" />. Otherwise answer in <answer>...</answer>."
        ),
        error_instruction: str = (
            "Emit a corrected <zoom_roi .../> if you still need visual detail. "
            "If the current evidence is enough, answer in <answer>...</answer>."
        ),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_zoom_calls = int(max_zoom_calls)
        self.min_relative_area = float(min_relative_area)
        self.max_relative_area = float(max_relative_area)
        self.continue_instruction = continue_instruction
        self.error_instruction = error_instruction

        # We use a custom zoom tag parser, not the generic OpenAI tool stack.
        self.tools = {}
        self.tool_schemas = []

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images") or []
        videos = multi_modal_data.get("videos")
        if videos:
            raise NotImplementedError("ActivePerceptionZoomAgentLoop currently supports image-only tasks.")
        if not images:
            raise ValueError("ActivePerceptionZoomAgentLoop requires at least one image in the prompt.")

        metrics = {}
        request_id = uuid4().hex
        extra_info = copy.deepcopy(dict(kwargs.get("extra_info") or {}))
        image_size = dict(extra_info.get("image_size") or {})
        if "width" not in image_size or "height" not in image_size:
            image_size["width"] = images[0].width
            image_size["height"] = images[0].height
            extra_info["image_size"] = image_size

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs={},
        )
        agent_data.extra_fields.update(
            {
                "active_tool_trace": [],
                "executed_observation_tokens": [],
                "sample_extra_info": extra_info,
                "tool_rewards": [],
                "turn_scores": [],
                "zoom_call_count": 0,
            }
        )
        agent_data.extra_fields["active_view_bboxes"] = [(0.0, 0.0, 1.0, 1.0) for _ in agent_data.image_data]

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                prompt_ids = await self.apply_chat_template(
                    agent_data.messages,
                    images=agent_data.image_data,
                    videos=agent_data.video_data,
                )
                agent_data.prompt_ids = prompt_ids
                state = AgentState.GENERATING
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                state = AgentState.TERMINATED

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        output_multi_modal = {"images": agent_data.image_data}
        if agent_data.video_data is not None:
            output_multi_modal["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=output_multi_modal,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        return output

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
    ) -> AgentState:
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        for key, value in output.extra_fields.items():
            if key == "max_global_steps" and value:
                agent_data.extra_fields[key] = value
            elif key not in agent_data.extra_fields:
                agent_data.extra_fields[key] = value

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        return AgentState.PROCESSING_TOOLS

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )
        parsed = parse_reasoning_trace(assistant_message)
        agent_data.messages.append({"role": "assistant", "content": assistant_message})

        current_view_bbox = agent_data.extra_fields["active_view_bboxes"][-1]
        latest_image = agent_data.image_data[-1]
        sample_extra_info = agent_data.extra_fields["sample_extra_info"]

        if parsed.zoom_calls:
            if agent_data.extra_fields["zoom_call_count"] >= self.max_zoom_calls:
                trace = zoom_limit_trace(
                    step_index=parsed.zoom_calls[0].step_index,
                    current_view_bbox_norm=current_view_bbox,
                    max_zoom_calls=self.max_zoom_calls,
                )
                crop = None
            else:
                trace, crop = execute_zoom_action(
                    image=latest_image,
                    current_view_bbox_norm=current_view_bbox,
                    zoom_call=parsed.zoom_calls[0],
                    extra_info=sample_extra_info,
                    min_relative_area=self.min_relative_area,
                    max_relative_area=self.max_relative_area,
                )
        elif parsed.errors:
            trace = malformed_zoom_trace(
                raw_tag=parsed.errors[0],
                step_index=len(agent_data.extra_fields["active_tool_trace"]) + 1,
                current_view_bbox_norm=current_view_bbox,
            )
            crop = None
        else:
            return AgentState.TERMINATED

        continue_instruction = self.continue_instruction if trace.status == STATUS_ZOOM_EXECUTED else self.error_instruction
        observation_message = build_observation_message(
            trace,
            continue_instruction=continue_instruction,
            image=crop,
        )
        agent_data.messages.append(observation_message)
        agent_data.extra_fields["active_tool_trace"].append(trace.to_dict())
        agent_data.extra_fields["tool_rewards"].append(float(trace.tool_reward))
        agent_data.extra_fields["zoom_call_count"] += 1
        if trace.observation_token:
            agent_data.extra_fields["executed_observation_tokens"].append(trace.observation_token)

        with simple_timer("tool_calls", agent_data.metrics):
            observation_ids = await self.apply_chat_template(
                [observation_message],
                images=[crop] if crop is not None else None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(observation_ids) >= self.response_length:
            return AgentState.TERMINATED

        if crop is not None:
            agent_data.image_data.append(crop)
            agent_data.extra_fields["active_view_bboxes"].append(trace.executed_bbox_norm)

        agent_data.prompt_ids += observation_ids
        agent_data.response_mask += [0] * len(observation_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(observation_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING
