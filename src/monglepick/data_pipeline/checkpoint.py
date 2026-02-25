"""
파이프라인 체크포인트 관리.

어디까지 수집/적재했는지 추적하여 중단 시 이어서 진행할 수 있게 한다.

체크포인트 파일: data/checkpoint.json
{
    "tmdb_api_loaded_ids": [11, 12, 13, ...],     # TMDB API로 수집하여 적재 완료된 ID
    "kaggle_loaded_ids": [862, 1893, ...],          # Kaggle CSV에서 적재 완료된 ID
    "embedded_ids": [11, 12, 862, ...],             # 임베딩 완료된 ID
    "failed_ids": [99999, ...],                     # 처리 실패한 ID
    "last_updated": "2026-02-25T14:30:00",
    "tmdb_api_total_collected": 3727,               # TMDB API 수집 시도 총 수
    "kaggle_total_available": 44176,                # Kaggle에서 보강 가능한 총 수
}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

CHECKPOINT_FILE = Path("data/checkpoint.json")


class PipelineCheckpoint:
    """파이프라인 진행 상태를 파일로 관리한다."""

    def __init__(self, filepath: Path = CHECKPOINT_FILE) -> None:
        self.filepath = filepath
        self.tmdb_api_loaded_ids: set[int] = set()
        self.kaggle_loaded_ids: set[int] = set()
        self.embedded_ids: set[int] = set()
        self.failed_ids: set[int] = set()
        self.last_updated: str = ""
        self.tmdb_api_total_collected: int = 0
        self.kaggle_total_available: int = 0

    def load(self) -> None:
        """체크포인트 파일에서 상태를 로드한다."""
        if self.filepath.exists():
            data = json.loads(self.filepath.read_text())
            self.tmdb_api_loaded_ids = set(data.get("tmdb_api_loaded_ids", []))
            self.kaggle_loaded_ids = set(data.get("kaggle_loaded_ids", []))
            self.embedded_ids = set(data.get("embedded_ids", []))
            self.failed_ids = set(data.get("failed_ids", []))
            self.last_updated = data.get("last_updated", "")
            self.tmdb_api_total_collected = data.get("tmdb_api_total_collected", 0)
            self.kaggle_total_available = data.get("kaggle_total_available", 0)
            logger.info(
                "checkpoint_loaded",
                tmdb_api=len(self.tmdb_api_loaded_ids),
                kaggle=len(self.kaggle_loaded_ids),
                embedded=len(self.embedded_ids),
                failed=len(self.failed_ids),
            )
        else:
            logger.info("checkpoint_not_found_creating_new")

    def save(self) -> None:
        """체크포인트 상태를 파일에 저장한다."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now().isoformat()
        data = {
            "tmdb_api_loaded_ids": sorted(self.tmdb_api_loaded_ids),
            "kaggle_loaded_ids": sorted(self.kaggle_loaded_ids),
            "embedded_ids": sorted(self.embedded_ids),
            "failed_ids": sorted(self.failed_ids),
            "last_updated": self.last_updated,
            "tmdb_api_total_collected": self.tmdb_api_total_collected,
            "kaggle_total_available": self.kaggle_total_available,
        }
        self.filepath.write_text(json.dumps(data))
        logger.info(
            "checkpoint_saved",
            tmdb_api=len(self.tmdb_api_loaded_ids),
            kaggle=len(self.kaggle_loaded_ids),
            total=len(self.all_loaded_ids),
        )

    @property
    def all_loaded_ids(self) -> set[int]:
        """적재 완료된 전체 ID (TMDB API + Kaggle)."""
        return self.tmdb_api_loaded_ids | self.kaggle_loaded_ids

    def summary(self) -> str:
        """현재 상태 요약 문자열."""
        total = len(self.all_loaded_ids)
        return (
            f"TMDB API: {len(self.tmdb_api_loaded_ids)} | "
            f"Kaggle: {len(self.kaggle_loaded_ids)} | "
            f"총 적재: {total} | "
            f"실패: {len(self.failed_ids)} | "
            f"최종 업데이트: {self.last_updated}"
        )
