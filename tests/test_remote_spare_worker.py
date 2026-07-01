import importlib.util
import os
from pathlib import Path
import subprocess
import time


def _load_remote_spare_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "racer_remote_spare_worker.py"
    spec = importlib.util.spec_from_file_location("racer_remote_spare_worker", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_spare_worker_expands_rank_ranges():
    module = _load_remote_spare_module()

    assert module._expand_int_list("0-3,5,7-6") == [0, 1, 2, 3, 5, 7, 6]
    assert module._expand_int_list("8") == [8]


def _run_script(
    script: str,
    tmp_path: Path,
    *,
    check: bool = True,
    **env_updates: str,
) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "MASTER_ADDR": "127.0.0.1",
            "OUTPUT_ROOT": str(tmp_path),
            "RACER_K": "6",
            "RACER_M": "2",
            "RACER_TRAIN_RANKS": "0-7",
            "RACER_SPARE_RANKS": "8",
            "RACER_RUNTIME_PORT": "29610",
            "PYTHONPATH": f"{root}:{env.get('PYTHONPATH', '')}",
        }
    )
    env.update(env_updates)
    return subprocess.run(
        ["bash", str(root / script)],
        cwd=str(root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
        timeout=30,
    )


def test_pai_multinode_dry_run_assigns_train_and_remote_spare_roles(tmp_path):
    common = {
        "MODE": "racer_pinned_remote_spare",
        "NNODES": "2",
        "NPROC_PER_NODE": "4",
    }

    node0 = _run_script("examples/pai_run_megatron_multinode.sh", tmp_path, NODE_RANK="0", **common)
    assert "NODE_ROLE=train" in node0.stdout
    assert "TORCHRUN_WORLD_SIZE=8" in node0.stdout
    assert "RACER_RUNTIME_WORLD_SIZE=9" in node0.stdout
    assert "RACER_CSD_LOCAL_RANKS=0-3" in node0.stdout
    assert "RACER_CSD_CHECKSUM_TYPE=sample64" in node0.stdout
    assert "RACER_CSD_MANIFEST_UPDATE_MODE=batch" in node0.stdout
    assert "CSD_NATIVE_PINNED_TOTAL_BYTES=103079215104" in node0.stdout

    node1 = _run_script("examples/pai_run_megatron_multinode.sh", tmp_path, NODE_RANK="1", **common)
    assert "NODE_ROLE=train" in node1.stdout
    assert "RACER_CSD_LOCAL_RANKS=4-7" in node1.stdout

    node2 = _run_script("examples/pai_run_megatron_multinode.sh", tmp_path, NODE_RANK="2", **common)
    assert "NODE_ROLE=spare" in node2.stdout
    assert "RACER_CSD_LOCAL_RANKS=8" in node2.stdout
    assert "CSD_NATIVE_PINNED_TOTAL_BYTES=0" in node2.stdout
    assert "DRY_RUN=1: 参数校验完成，跳过 remote spare worker。" in node2.stdout


def test_pai_multinode_check_paths_accepts_indexed_dataset_prefix(tmp_path):
    fake_megatron = tmp_path / "Megatron-LM-FT"
    fake_megatron.mkdir()
    (fake_megatron / "pretrain_gpt.py").write_text("# fake pretrain entry\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_torchrun.chmod(0o755)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_prefix = data_dir / "my_shakespeare_text_document"
    (data_prefix.with_suffix(".bin")).write_bytes(b"fake-bin")
    (data_prefix.with_suffix(".idx")).write_bytes(b"fake-idx")
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    vocab_file.write_text("{}", encoding="utf-8")
    merges_file.write_text("#version: 0.2\n", encoding="utf-8")

    result = _run_script(
        "examples/pai_run_megatron_multinode.sh",
        tmp_path,
        MODE="baseline",
        NODE_RANK="0",
        NNODES="1",
        NPROC_PER_NODE="1",
        DRY_RUN="0",
        MEGATRON_ROOT=str(fake_megatron),
        DATA_PATH=str(data_prefix),
        GPT2_VOCAB_FILE=str(vocab_file),
        GPT2_MERGE_FILE=str(merges_file),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert not data_prefix.exists()
    assert "torchrun exit code: 0" in result.stdout


def test_pai_multinode_dry_run_accepts_default_egm_remote_spare_factory(tmp_path):
    result = _run_script(
        "examples/pai_run_megatron_multinode.sh",
        tmp_path,
        MODE="racer_egm_remote_spare",
        NODE_RANK="0",
        NNODES="2",
        NPROC_PER_NODE="4",
    )

    assert "backend=egm" in result.stdout
    assert "MODE=racer_egm_remote_spare" in result.stdout
    assert "RACER_SPARE_LAUNCH_MODE=remote" in result.stdout
    assert "CSD_EGM_RUNTIME_FACTORY=racer.egm_runtime:create_runtime" in result.stdout


def test_pai_multinode_dry_run_rejects_noncanonical_remote_spare_rank(tmp_path):
    result = _run_script(
        "examples/pai_run_megatron_multinode.sh",
        tmp_path,
        check=False,
        MODE="racer_pinned_remote_spare",
        NODE_RANK="0",
        NNODES="2",
        NPROC_PER_NODE="4",
        RACER_SPARE_RANKS="9",
    )

    assert result.returncode != 0
    assert "remote spare rank 必须紧跟 train rank 连续编号" in result.stdout
    assert "RACER_SPARE_RANKS=8" in result.stdout


def test_builtin_egm_runtime_factory_entrypoint_is_importable():
    from racer.csd import _load_object

    factory = _load_object("racer.egm_runtime:create_runtime")

    assert callable(factory)


def test_pai_multinode_dry_run_supports_single_node_local_spare(tmp_path):
    result = _run_script(
        "examples/pai_run_megatron_multinode.sh",
        tmp_path,
        MODE="racer_pinned_single_node",
        NODE_RANK="0",
        NNODES="1",
        NPROC_PER_NODE="4",
        RACER_K="3",
        RACER_M="1",
        RACER_TRAIN_RANKS="0-3",
        RACER_SPARE_RANKS="4",
    )

    assert "NODE_ROLE=train" in result.stdout
    assert "TORCHRUN_WORLD_SIZE=4" in result.stdout
    assert "RACER_RUNTIME_WORLD_SIZE=5" in result.stdout
    assert "RACER_SPARE_LAUNCH_MODE=local" in result.stdout
    assert "RACER_SPARE_RANKS=4" in result.stdout


def test_pai_restart_driver_dry_run_uses_persistent_then_existing_csd(tmp_path):
    result = _run_script(
        "examples/pai_run_megatron_restart_driver.sh",
        tmp_path,
        MODE="racer_pinned_remote_spare",
        NODE_RANK="0",
        NNODES="2",
        NPROC_PER_NODE="4",
        BASE_RUN_ID="pytest_restart_driver",
        RESTART_OVERWRITE="1",
        PAI_TOTAL_NODES="1",
    )

    assert "FINAL_TRAIN_ITERS=80" in result.stdout
    phase0 = tmp_path / "logs" / "pytest_restart_driver_phase00.driver.node0.log"
    phase1 = tmp_path / "logs" / "pytest_restart_driver_phase01.driver.node0.log"
    summary = tmp_path / "restart_state" / "pytest_restart_driver" / "summary.md"
    assert phase0.exists()
    assert phase1.exists()
    assert summary.exists()
    assert "RACER_CSD_MODE=persistent" in phase0.read_text(encoding="utf-8")
    assert "RACER_CSD_MODE=existing" in phase1.read_text(encoding="utf-8")


def test_pai_restart_driver_dry_run_passes_egm_runtime_settings_to_phases(tmp_path):
    result = _run_script(
        "examples/pai_run_megatron_restart_driver.sh",
        tmp_path,
        MODE="racer_egm_remote_spare",
        NODE_RANK="0",
        NNODES="2",
        NPROC_PER_NODE="4",
        BASE_RUN_ID="pytest_restart_driver_egm",
        RESTART_OVERWRITE="1",
        PAI_TOTAL_NODES="1",
        CSD_EGM_NUMA_ID="7",
        CSD_EGM_ACCESSING_DEVICES="0,1,2,3",
        CSD_EGM_POOL_ID="pool-a",
        CSD_EGM_OWNER_NODE="node-a",
        CSD_EGM_OWNER_TRAY="tray-a",
        CSD_EGM_HOME_DEVICE="2",
        CSD_NATIVE_PINNED_DEVICE="3",
        RACER_BUFFER_SIZE="536870912",
        RACER_PAYLOAD_POOL_PREWARM_CHUNKS="2",
    )

    assert "FINAL_TRAIN_ITERS=80" in result.stdout
    phase0 = tmp_path / "logs" / "pytest_restart_driver_egm_phase00.driver.node0.log"
    config_path = tmp_path / "restart_state" / "pytest_restart_driver_egm" / "config.txt"
    assert phase0.exists()
    text = phase0.read_text(encoding="utf-8")
    assert "MODE=racer_egm_remote_spare" in text
    assert "CSD_EGM_RUNTIME_FACTORY=racer.egm_runtime:create_runtime" in text
    assert "RACER_CSD_CHECKSUM_TYPE=sample64" in text
    assert "RACER_CSD_MANIFEST_UPDATE_MODE=batch" in text
    assert "CSD_EGM_POOL_ID=pool-a" in text
    assert "CSD_EGM_OWNER_NODE=node-a" in text
    assert "CSD_EGM_OWNER_TRAY=tray-a" in text
    assert "CSD_EGM_HOME_DEVICE=2" in text
    assert "CSD_EGM_NUMA_ID=7" in text
    assert "CSD_EGM_ACCESSING_DEVICES=0,1,2,3" in text
    assert "CSD_NATIVE_PINNED_DEVICE=3" in text
    config = config_path.read_text(encoding="utf-8")
    assert "RACER_BUFFER_SIZE=536870912" in config
    assert "RACER_PAYLOAD_POOL_PREWARM_CHUNKS=2" in config
    assert "CSD_EGM_POOL_ID=pool-a" in config
    assert "CSD_EGM_HOME_DEVICE=2" in config
    assert "CSD_EGM_NUMA_ID=7" in config


def test_pai_restart_driver_dry_run_coordinates_three_nodes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    base_env = dict(os.environ)
    base_env.update(
        {
            "DRY_RUN": "1",
            "MODE": "racer_pinned_remote_spare",
            "MASTER_ADDR": "127.0.0.1",
            "OUTPUT_ROOT": str(tmp_path),
            "BASE_RUN_ID": "pytest_restart_driver_3nodes",
            "RESTART_OVERWRITE": "1",
            "NNODES": "2",
            "NPROC_PER_NODE": "4",
            "PAI_TOTAL_NODES": "3",
            "RACER_K": "6",
            "RACER_M": "2",
            "RACER_TRAIN_RANKS": "0-7",
            "RACER_SPARE_RANKS": "8",
            "MARKER_POLL_SECONDS": "1",
            "PHASE_TIMEOUT_SECONDS": "60",
            "PYTHONPATH": f"{root}:{base_env.get('PYTHONPATH', '')}",
        }
    )

    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for node_rank in (1, 2, 0):
        env = dict(base_env)
        env["NODE_RANK"] = str(node_rank)
        proc = subprocess.Popen(
            ["bash", str(root / "examples" / "pai_run_megatron_restart_driver.sh")],
            cwd=str(root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append((node_rank, proc))
        time.sleep(0.2)

    outputs: dict[int, str] = {}
    try:
        for node_rank, proc in processes:
            stdout, _stderr = proc.communicate(timeout=60)
            outputs[node_rank] = stdout
            assert proc.returncode == 0, f"node{node_rank} failed:\n{stdout}"
    finally:
        for _node_rank, proc in processes:
            if proc.poll() is None:
                proc.kill()

    state_dir = tmp_path / "restart_state" / "pytest_restart_driver_3nodes"
    assert (state_dir / "summary.md").exists()
    config = (state_dir / "config.txt").read_text(encoding="utf-8")
    assert "PAI_TOTAL_NODES=3" in config
    for phase in range(4):
        phase_name = f"phase{phase:02d}"
        for node_rank in range(3):
            assert (state_dir / f"{phase_name}.node{node_rank}.done").exists()

    node2_phase0 = tmp_path / "logs" / "pytest_restart_driver_3nodes_phase00.driver.node2.log"
    assert node2_phase0.exists()
    node2_text = node2_phase0.read_text(encoding="utf-8")
    assert "NODE_ROLE=spare" in node2_text
    assert "RACER_CSD_LOCAL_RANKS=8" in node2_text
    assert "CSD_NATIVE_PINNED_TOTAL_BYTES=0" in node2_text


def test_pai_multinode_dry_run_uses_larger_default_pool_for_5_3b_train_nodes(tmp_path):
    result = _run_script(
        "examples/pai_run_megatron_multinode.sh",
        tmp_path,
        MODE="racer_pinned_remote_spare",
        MODEL_SIZE="5.3b",
        NODE_RANK="0",
        NNODES="2",
        NPROC_PER_NODE="4",
    )

    assert "CSD_NATIVE_PINNED_TOTAL_BYTES=274877906944" in result.stdout
