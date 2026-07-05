# pyright: reportUnusedImport = false

from collections.abc import Mapping, Sequence

from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.tasks import (
    CancelTask,
    ConnectToGroup,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import Base64Image, Base64ImageHash
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgress,
)
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.runner.supervisor import RunnerSupervisor


def plan(
    node_id: NodeId,
    # Runners is expected to be FRESH and so should not come from state
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    instances: Mapping[InstanceId, Instance],
    all_runners: Mapping[RunnerId, RunnerStatus],  # all global
    tasks: Mapping[TaskId, Task],
    input_chunk_buffer: Mapping[CommandId, Mapping[int, InputImageChunk]],
    image_cache: Mapping[Base64ImageHash, Base64Image],
    instance_backoff: KeyedBackoff[InstanceId],
    download_backoff: KeyedBackoff[ModelId],
) -> Task | None:
    # Python short circuiting OR logic should evaluate these sequentially.
    return (
        _kill_runner(runners, all_runners, instances)
        or _cancel_tasks(runners, tasks)
        or _create_runner(node_id, runners, all_runners, instances, instance_backoff)
        or _model_needs_download(
            node_id, runners, global_download_status, download_backoff
        )
        or _init_distributed_backend(runners, all_runners)
        or _load_model(runners, all_runners, global_download_status)
        or _ready_to_warmup(runners, all_runners)
        or _pending_tasks(runners, tasks, all_runners, input_chunk_buffer, image_cache)
    )


def _kill_runner(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
) -> Shutdown | None:
    for runner in runners.values():
        runner_id = runner.bound_instance.bound_runner_id
        if (instance_id := runner.bound_instance.instance.instance_id) not in instances:
            return Shutdown(instance_id=instance_id, runner_id=runner_id)
        if isinstance(runner.status, RunnerFailed):
            return Shutdown(
                instance_id=runner.bound_instance.instance.instance_id,
                runner_id=runner_id,
            )

        for (
            global_runner_id
        ) in runner.bound_instance.instance.shard_assignments.node_to_runner.values():
            if runner_id == global_runner_id:
                continue

            if isinstance(all_runners.get(global_runner_id, None), RunnerFailed):
                return Shutdown(
                    instance_id=instance_id,
                    runner_id=runner_id,
                )


def _create_runner(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
    instance_backoff: KeyedBackoff[InstanceId],
) -> CreateRunner | None:
    for instance in instances.values():
        runner_id = instance.shard_assignments.node_to_runner.get(node_id, None)
        if runner_id is None:
            continue

        if runner_id in runners:
            continue

        # don't create runners if any other nodes have runners that have failed - wait for them to fix themselves first.
        instance_has_failed_runner = any(
            isinstance(all_runners.get(remote_runner_id), RunnerFailed)
            for remote_runner_id in instance.shard_assignments.node_to_runner.values()
            if remote_runner_id != runner_id
        )
        we_have_failed_before = isinstance(all_runners.get(runner_id), RunnerFailed)
        if instance_has_failed_runner and not we_have_failed_before:
            continue

        if not instance_backoff.should_proceed(instance.instance_id):
            continue

        return CreateRunner(
            instance_id=instance.instance_id,
            bound_instance=BoundInstance(
                instance=instance, bound_runner_id=runner_id, bound_node_id=node_id
            ),
        )


def _model_needs_download(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    download_backoff: KeyedBackoff[ModelId],
) -> DownloadModel | None:
    local_downloads = global_download_status.get(node_id, [])
    download_status = {
        dp.shard_metadata.model_card.model_id: dp for dp in local_downloads
    }

    for runner in runners.values():
        model_id = runner.bound_instance.bound_shard.model_card.model_id
        if (
            isinstance(runner.status, RunnerIdle)
            and (
                model_id not in download_status
                or not isinstance(
                    download_status[model_id],
                    (DownloadOngoing, DownloadCompleted, DownloadFailed),
                )
            )
            and download_backoff.should_proceed(model_id)
        ):
            # We don't invalidate download_status randomly in case a file gets deleted on disk
            return DownloadModel(
                instance_id=runner.bound_instance.instance.instance_id,
                shard_metadata=runner.bound_instance.bound_shard,
            )


def _init_distributed_backend(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
):
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        is_single_node_instance = len(shard_assignments.runner_to_shard) == 1
        if is_single_node_instance:
            continue

        # A runner fires ConnectToGroup as soon as its own runner is idle AND every
        # peer runner in the group has been created (i.e. has any known status).
        #
        # Previously this required rank N-1 to wait until all peers had reached
        # RunnerConnecting. That created a fragile inter-rank status dependency:
        # the RunnerConnecting status update is emitted from inside the runner
        # subprocess immediately before it enters the blocking generator.connect()
        # call, and in practice it does not always propagate to peers before they
        # evaluate plan(). The result was a deterministic deadlock: rank 0 entered
        # connect() and blocked waiting for rank 1 to join the collective, while
        # rank 1's plan() looped forever waiting for rank 0's RunnerConnecting.
        #
        # Rank coordination/ordering is handled inside generator.connect() (the
        # JACCL/QP layer synchronises ranks via the coordinator regardless of the
        # order in which ranks call connect), so the plan layer does not need to
        # enforce it. We only require that all peers exist so connect() has someone
        # to rendezvous with.
        if not isinstance(runner.status, RunnerIdle):
            continue

        if not all(
            all_runners.get(global_runner_id) is not None
            for global_runner_id in shard_assignments.runner_to_shard
        ):
            continue

        return ConnectToGroup(instance_id=instance.instance_id)

    return None


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> LoadModel | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        all_local_downloads_complete = all(
            nid in global_download_status
            and any(
                isinstance(dp, DownloadCompleted)
                and dp.shard_metadata.model_card.model_id == shard_assignments.model_id
                for dp in global_download_status[nid]
            )
            for nid in shard_assignments.node_to_runner
        )
        if not all_local_downloads_complete:
            continue

        is_single_node_instance = len(instance.shard_assignments.runner_to_shard) == 1
        if is_single_node_instance and isinstance(runner.status, RunnerIdle):
            return LoadModel(instance_id=instance.instance_id)

        # Local status is reliable (set in-process by the supervisor); peer status is
        # propagated through the master event log and is not reliably visible to us
        # (see note in _init_distributed_backend). Once our runner is connected and
        # every peer runner exists in the group, proceed to load — the JACCL group is
        # already established by ConnectToGroup, and load() does not require peer
        # coordination beyond the group existing.
        if isinstance(runner.status, RunnerConnected) and all(
            all_runners.get(global_runner_id) is not None
            for global_runner_id in shard_assignments.runner_to_shard
        ):
            return LoadModel(instance_id=instance.instance_id)

    return None


def _ready_to_warmup(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> StartWarmup | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments
        shard = runner.bound_instance.bound_shard
        device_rank = shard.device_rank
        runner_id = runner.bound_instance.bound_runner_id
        world_size = shard.world_size

        # Warmup (prefill) is itself a distributed collective: the MLX/JACCL layer
        # synchronises ranks internally regardless of who starts first. We therefore
        # fire StartWarmup as soon as our runner is locally loaded and every peer
        # runner exists. The previous rank-ordered gate depended on peer
        # Loaded/WarmingUp status, which is not reliably propagated (see note in
        # _init_distributed_backend) and deadlocked warmup.
        if isinstance(runner.status, RunnerLoaded) and all(
            all_runners.get(global_runner_id) is not None
            for global_runner_id in shard_assignments.runner_to_shard
        ):
            return StartWarmup(instance_id=instance.instance_id)

    return None


def _pending_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
    all_runners: Mapping[RunnerId, RunnerStatus],
    input_chunk_buffer: Mapping[CommandId, Mapping[int, InputImageChunk]],
    image_cache: Mapping[Base64ImageHash, Base64Image],
) -> Task | None:
    for task in tasks.values():
        # for now, just forward chat completions
        # TODO(ciaran): do this better!
        if not isinstance(task, (TextGeneration, ImageGeneration, ImageEdits)):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue

        if isinstance(task, ImageEdits) and task.task_params.total_input_chunks > 0:
            received = len(input_chunk_buffer.get(task.command_id, {}))
            if received < task.task_params.total_input_chunks:
                continue  # Wait for all chunks to arrive

        if (
            isinstance(task, TextGeneration)
            and task.task_params.image_hashes
            and not all(
                h in image_cache for h in task.task_params.image_hashes.values()
            )
        ):
            continue  # Wait for all images to be assembled into the cache

        for runner in runners.values():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue

            # the task status _should_ be set to completed by the LAST runner
            # it is currently set by the first
            # this is definitely a hack
            if task.task_id in runner.completed or task.task_id in runner.in_progress:
                continue

            # Dispatch based on LOCAL runner status (reliable; set in-process by the
            # supervisor) plus peer-runner existence. The previous gate also required
            # every peer runner to be observed as Ready/Running, but peer status is not
            # reliably propagated (see note in _init_distributed_backend), so only one
            # rank would pass this gate and receive the TextGeneration task. The other
            # rank never stepped, so the decode all-gather/send-recv collective in the
            # sharded forward pass deadlocked and no tokens were produced.
            if isinstance(runner.status, (RunnerReady, RunnerRunning)) and all(
                all_runners.get(global_runner_id) is not None
                for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
            ):
                return task


def _cancel_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
) -> Task | None:
    for task in tasks.values():
        if task.task_status != TaskStatus.Cancelled:
            continue
        for runner_id, runner in runners.items():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue
            if task.task_id in runner.cancelled:
                continue
            return CancelTask(
                instance_id=task.instance_id,
                cancelled_task_id=task.task_id,
                runner_id=runner_id,
            )
