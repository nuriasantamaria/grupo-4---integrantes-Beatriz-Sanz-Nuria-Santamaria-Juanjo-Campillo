"""Hilos de trabajo: previsualización y análisis (Richardson, convergencia) fuera del hilo de la interfaz"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from PySide6 import QtCore

from simulation import run_each, run_preview, SimConfig


class PreviewWorker(QtCore.QThread):
    """Calcula las previsualizaciones fuera del hilo de la interfaz"""

    done = QtCore.Signal(int, object, float)       # token, {k: preview}, segundos

    def __init__(self, token: int, jobs: list[tuple[int, SimConfig]], T: float) -> None:
        super().__init__()
        self.token, self.jobs, self.T = token, jobs, T
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        """Ejecuta las previsualizaciones y emite el resultado salvo que se cancele"""
        t0 = time.perf_counter()
        out = {}
        for k, cfg in self.jobs:
            try:
                r = run_preview(cfg, self.T, lambda: self._cancel)
            except Exception as exc:            # el hilo nunca debe morir en silencio
                r = {"t": np.zeros(1), "W": np.zeros((1, 3)), "dT": np.zeros(1), "dL": np.zeros(1),
                     "r": np.zeros(1), "psi": np.zeros(1), "lag": np.zeros(1), "diverged": True,
                     "error": f"{type(exc).__name__}: {exc}", "truncated": False}
            if r is None:
                return
            out[k] = r
        self.done.emit(self.token, out, time.perf_counter() - t0)


class AnalysisWorker(QtCore.QThread):
    """Calcula un análisis pesado de cada simulación fuera del hilo de la interfaz"""

    done = QtCore.Signal(str, object, float)       # tipo de análisis, {k: resultado o excepción}, segundos

    def __init__(self, kind: str, cfgs: list[SimConfig], job: Callable[[SimConfig], Any]) -> None:
        super().__init__()
        self.kind, self.cfgs, self.job = kind, cfgs, job
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        """Ejecuta el análisis y emite el resultado salvo que se cancele"""
        t0 = time.perf_counter()
        out = run_each(self.cfgs, self.job, lambda: self._cancel)
        if not self._cancel:
            self.done.emit(self.kind, out, time.perf_counter() - t0)
