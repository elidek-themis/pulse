import json
import logging

from enum import StrEnum
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


class FileStatus(StrEnum):
    OK = "ok"
    EXISTS = "File name already exists"
    INVALID_SCHEMA = "Invalid schema (required fields: 'A', 'B', 'alias')."
    MISSING_VALUES = "File contains missing values."


@dataclass
class PulseFile:
    path: Path
    df: pd.DataFrame

    def __post_init__(self):
        self.name = self.path.name
        self.stem = self.path.stem
        self.suffix = self.path.suffix

    def to_dict(self, orient: str | None) -> dict:
        return self.df.to_dict(orient=orient)

    def __repr__(self) -> str:
        data = {
            "name": self.name,
            "rows": len(self.df),
            "cols": len(self.df.columns),
        }

        return json.dumps(obj=data, indent=2)


class FileManager:
    SUFFIXES = ("*.json", "*.csv")

    def __init__(self, dir: Path, schema: set = None):
        self.root = dir
        self.schema = schema if schema else set()

        files = [f for suffix in self.SUFFIXES for f in dir.glob(suffix)]

        self.data: dict[str, PulseFile] = {}
        for file in files:
            df = pd.read_json(file) if file.suffix == ".json" else pd.read_csv(file)
            self._add(file=file, df=df)

    def add(self, name: str, df: pd.DataFrame) -> FileStatus:
        file = self.root.joinpath(name).with_suffix(".json")
        status = self._add(file=file, df=df)
        if status == FileStatus.OK:
            self[name].df.to_json(file, orient="records", indent=2)
        return status

    def delete(self, name: str):
        if pulse_file := self.data[name]:
            # remove from disk
            pulse_file.path.unlink(missing_ok=True)
            # update data
            del self.data[name]

    def update(self, name: str, df: pd.DataFrame):
        if pulse_file := self.data[name]:
            # update disk file
            if pulse_file.suffix == ".json":
                df.to_json(pulse_file.path)
            else:
                df.to_csv(pulse_file.path, index=False)
            # update data
            pulse_file.data = df

    def _add(self, file: Path, df: pd.DataFrame) -> FileStatus:
        if file.stem in self.data:
            logger.info(f"File {file.stem} already exists.")
            return FileStatus.EXISTS

        if not self._is_valid_schema(df=df):
            logger.info(f"File {file.stem} has invalid schema.")
            return FileStatus.INVALID_SCHEMA

        if self._has_nan(df=df):
            logger.info(f"File {file.stem} contains missing values.")
            return FileStatus.MISSING_VALUES

        pulse_file = PulseFile(path=file, df=df)
        self.data[pulse_file.stem] = pulse_file

        return FileStatus.OK

    def _is_valid_schema(self, df: pd.DataFrame) -> bool:
        return self.schema.issubset(df.columns)

    def _has_nan(self, df: pd.DataFrame) -> bool:
        return df.isnull().any().any()

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> PulseFile:
        return self.data[key]
