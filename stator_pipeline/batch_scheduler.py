"""batch_scheduler.py — Pure Python batch scheduler using multiprocessing."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

from .params import StatorParams
from .export_engine import ExportConfig, ExportFormat, outputs_exist


@dataclass
class BatchJob:
    job_id: str
    params: StatorParams
    export_config: ExportConfig


@dataclass
class BatchConfig:
    max_parallel: int = 0
    skip_existing: bool = True
    job_timeout_sec: int = 300
    write_summary: bool = True


@dataclass
class BatchResult:
    job_id: str = ""
    success: bool = False
    error: str = ""
    msh_path: str = ""
    vtk_path: str = ""
    hdf5_path: str = ""
    json_path: str = ""


def _execute_job(job: BatchJob) -> BatchResult:
    """Run one job in a subprocess. Returns BatchResult."""
    from .gmsh_backend import make_default_backend
    from .geometry_builder import GeometryBuilder
    from .topology_registry import TopologyRegistry
    from .mesh_generator import MeshGenerator
    from .export_engine import ExportEngine, compute_stem

    result = BatchResult(job_id=job.job_id)
    try:
        backend = make_default_backend()
        stem = compute_stem(job.params)
        backend.initialize(f"stator_{stem}")

        builder = GeometryBuilder(backend)
        geo = builder.build(job.params)
        if not geo.success:
            result.error = f"Geometry build failed: {geo.error_message}"
            return result

        registry = TopologyRegistry(job.params.n_slots)
        generator = MeshGenerator(backend)
        mesh = generator.generate(job.params, geo, registry)
        if not mesh.success:
            result.error = f"Mesh generation failed: {mesh.error_message}"
            return result

        os.makedirs(job.export_config.output_dir, exist_ok=True)
        engine = ExportEngine(backend)
        export_results = engine.write_all(job.params, mesh, job.export_config)

        backend.finalize()

        for r in export_results:
            if not r.success:
                continue
            if ExportFormat.MSH in r.format:
                result.msh_path = r.path
            elif ExportFormat.VTK in r.format:
                result.vtk_path = r.path
            elif ExportFormat.HDF5 in r.format:
                result.hdf5_path = r.path
            elif ExportFormat.JSON in r.format:
                result.json_path = r.path

        result.success = True
    except Exception as e:
        result.error = str(e)
    return result


class BatchScheduler:
    def __init__(self) -> None:
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(
        self,
        jobs: list[BatchJob],
        config: BatchConfig,
        progress_callback: Optional[Callable[[int, int, bool, str], None]] = None,
    ) -> list[BatchResult]:
        if not jobs:
            return []
        self._cancel = False

        max_workers = config.max_parallel or os.cpu_count() or 1
        results: list[BatchResult] = []
        jobs_done = 0
        total = len(jobs)

        # Split into skip-existing and to-run
        run_jobs: list[BatchJob] = []
        for job in jobs:
            if config.skip_existing and outputs_exist(job.params, job.export_config):
                r = BatchResult(job_id=job.job_id, success=True)
                results.append(r)
                jobs_done += 1
                if progress_callback:
                    progress_callback(jobs_done, total, True, job.job_id)
            else:
                run_jobs.append(job)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(_execute_job, j): j for j in run_jobs}
            for future in as_completed(future_to_job):
                if self._cancel:
                    break
                r = future.result()
                results.append(r)
                jobs_done += 1
                if progress_callback:
                    progress_callback(jobs_done, total, r.success, r.job_id)

        if config.write_summary and jobs:
            output_dir = jobs[0].export_config.output_dir
            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, "batch_summary.json")
            with open(summary_path, "w") as f:
                json.dump([
                    {"job_id": r.job_id, "success": r.success, "error": r.error, "msh_path": r.msh_path}
                    for r in results
                ], f, indent=2)

        return results
