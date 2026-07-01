import subprocess
import sys

import racer.csd as csd
import racer.egm_runtime as egm


class _FakeCudaFn:
    def __init__(self, impl):
        self.impl = impl

    def __call__(self, *args):
        return self.impl(*args)


class _FakeCudart:
    def __init__(self):
        self.created_props = None
        self.destroyed_pool = None
        self.cudaMemPoolCreate = _FakeCudaFn(self._mem_pool_create)
        self.cudaMemPoolSetAccess = _FakeCudaFn(lambda *_args: 0)
        self.cudaMallocFromPoolAsync = _FakeCudaFn(lambda *_args: 0)
        self.cudaFreeAsync = _FakeCudaFn(lambda *_args: 0)
        self.cudaMemPoolDestroy = _FakeCudaFn(self._mem_pool_destroy)

    def _mem_pool_create(self, pool_ptr, props_ptr):
        self.created_props = props_ptr._obj
        pool_ptr._obj.value = 0xC5D
        return 0

    def _mem_pool_destroy(self, pool):
        self.destroyed_pool = int(pool.value)
        return 0


def test_builtin_egm_runtime_uses_host_numa_pool(monkeypatch):
    fake = _FakeCudart()
    monkeypatch.setenv("RACER_CSD_PREWARM_CUDA_CONTEXTS", "0")
    monkeypatch.setattr(egm, "_configure_cudart_mempool_api", lambda: fake)
    monkeypatch.setattr(egm, "_load_cudart", lambda: fake)
    monkeypatch.setattr(egm, "_detect_host_numa_id", lambda device: 7)
    monkeypatch.setattr(egm, "_visible_cuda_device_count", lambda: 4)
    monkeypatch.setattr(csd, "_load_cudart", lambda: fake)
    monkeypatch.setattr(csd, "_cuda_set_device", lambda device: None)
    monkeypatch.setattr(csd.torch.cuda, "is_available", lambda: True)

    runtime = egm.create_runtime(home_device=2, total_bytes=0, segment_bytes=4096)

    assert fake.created_props is not None
    assert fake.created_props.allocType == 1
    assert fake.created_props.location.type == 3
    assert fake.created_props.location.id == 7
    assert runtime.numa_id == 7
    assert runtime.accessing_devices == (0, 1, 2, 3)
    caps = runtime.capabilities()
    assert caps["supports_egm_native_transport"] is True
    assert caps["uses_cuda_mempool"] is True
    assert caps["uses_cudaHostAlloc"] is False


def test_racer_package_lazily_exports_csd_symbols():
    code = (
        "import sys, racer; "
        "print('before', 'racer.csd' in sys.modules); "
        "print(racer.CheckpointStorageDaemonClient.__name__); "
        "print('after', 'racer.csd' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=15,
    )

    assert "before False" in result.stdout
    assert "CheckpointStorageDaemonClient" in result.stdout
    assert "after True" in result.stdout


def test_csd_module_entrypoint_does_not_preimport_itself():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "racer.csd",
            "--backend",
            "egm",
            "--egm-runtime-factory",
            "no.such:factory",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=15,
    )

    assert result.returncode != 0
    assert "RuntimeWarning" not in result.stdout
    assert "No module named 'no'" in result.stdout
