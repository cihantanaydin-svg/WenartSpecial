"""Service container shared by the API and workers."""

from __future__ import annotations

from dataclasses import dataclass, field

from archrender.core.cas import ProjectStore
from archrender.core.config import Settings
from archrender.db.database import Database
from archrender.models.license_gate import LicenseGate
from archrender.models.manager import ModelManager
from archrender.models.profiles import HardwareProfile
from archrender.models.registry import Registry
from archrender.pipeline.queue import JobQueue
from archrender.qa.runner import QAConfig
from archrender.render.blender import BlenderRunner
from archrender.scene.materials import MaterialLibrary


@dataclass
class Services:
    settings: Settings
    db: Database
    queue: JobQueue
    registry: Registry
    gate: LicenseGate
    profile: HardwareProfile
    library: MaterialLibrary
    qa_config: QAConfig
    blender: BlenderRunner
    _models: ModelManager | None = field(default=None, repr=False)

    @classmethod
    def create(cls, settings: Settings, *, migrate: bool = True) -> Services:
        settings.ensure_dirs()
        db = Database(settings.db_path)
        if migrate:
            db.migrate()
        cfg = settings.configs_dir
        return cls(
            settings=settings,
            db=db,
            queue=JobQueue(db),
            registry=Registry.load(cfg),
            gate=LicenseGate.load(cfg),
            profile=HardwareProfile.load(cfg, settings.profile),
            library=MaterialLibrary.load(cfg),
            qa_config=QAConfig.load(cfg),
            blender=BlenderRunner(settings),
        )

    @property
    def models(self) -> ModelManager:
        if self._models is None:
            self._models = ModelManager(self.profile, self.registry, self.gate)
        return self._models

    def store(self, project_id: str) -> ProjectStore:
        return ProjectStore(self.settings.projects_dir(), project_id)
