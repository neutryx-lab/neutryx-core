import jax
import jax.numpy as jnp

from neutryx.core.execution import MeshConfig, mesh_pjit, mesh_xmap
from neutryx.core.infrastructure.cluster import ClusterConfig


def _single_axis_mesh() -> MeshConfig:
    devices = jax.devices()
    return MeshConfig(shape=(len(devices),), axis_names=("data",))


def test_mesh_pjit_runs_on_available_devices():
    mesh = _single_axis_mesh()

    def fn(x):
        return x + 1

    compiled = mesh_pjit(fn, mesh=mesh)
    result = compiled(jnp.array(1.0))
    assert float(result) == 2.0


def test_mesh_xmap_binds_axis_resources():
    mesh = _single_axis_mesh()

    def fn(x):
        return x * 2

    mapped = mesh_xmap(
        fn,
        mesh=mesh,
        in_axes=("data",),
        out_axes="data",
        axis_resources={"data": "data"},
    )

    data = jnp.arange(mesh.shape_tuple[0], dtype=jnp.float32)
    result = mapped(data)
    assert result.shape == data.shape
    assert jnp.allclose(result, data * 2)


def test_cluster_config_env_roundtrip(monkeypatch):
    monkeypatch.setenv("JAX_CLUSTER_COORDINATOR_ADDRESS", "localhost")
    monkeypatch.setenv("JAX_CLUSTER_COORDINATOR_PORT", "1234")
    monkeypatch.setenv("JAX_CLUSTER_NUM_PROCESSES", "2")
    monkeypatch.setenv("JAX_CLUSTER_PROCESS_ID", "0")
    monkeypatch.setenv("JAX_CLUSTER_LOCAL_DEVICE_IDS", "0,1")
    monkeypatch.setenv("JAX_CLUSTER_MESH_AXIS_NAMES", "data")
    monkeypatch.setenv("JAX_CLUSTER_MESH_SHAPE", "2")

    cfg = ClusterConfig.from_env()
    assert cfg is not None
    assert cfg.coordinator() == "localhost:1234"
    assert cfg.mesh is not None
    assert cfg.mesh.shape_tuple == (2,)
    assert cfg.local_device_ids == (0, 1)


def test_cluster_config_initialize(monkeypatch):
    recorded = {}

    monkeypatch.setenv("JAX_USE_PJRT_C_API_ON_PLUGIN", "0")

    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: False)

    def fake_init(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(jax.distributed, "initialize", fake_init)

    cfg = ClusterConfig(
        coordinator_address="localhost",
        coordinator_port=9999,
        num_processes=2,
        process_id=1,
        local_device_ids=(0,),
    )

    cfg.initialize()

    assert recorded["coordinator_address"] == "localhost:9999"
    assert recorded["num_processes"] == 2
    assert recorded["process_id"] == 1
    assert recorded["local_device_ids"] == (0,)

