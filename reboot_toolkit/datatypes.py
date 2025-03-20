from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os
from functools import cached_property


def get_s3_bucket(org_id: Optional[str] = None) -> Optional[str]:
    """
    Get the S3 bucket from environment variable REBOOT_S3_BUCKET.
    If that isn't available and org_id is provided, return the reboot motion-provided bucket.
    If org_id is not provided, check environment variable ORG_ID for the org id.
    If org_id is still unavailalbe, return None.
    """
    if reboot_s3_bucket := os.getenv("REBOOT_S3_BUCKET"):
        return reboot_s3_bucket
    elif org_id:
        return f"reboot-motion-{org_id}"
    elif org_id := os.getenv("ORG_ID"):
        return f"reboot-motion-{org_id}"
    else:
        return None


class MocapType(str, Enum):
    HAWKEYE = "hawkeye"
    HAWKEYE_HFR = "hawkeyehfr"
    REBOOT_MOTION = "rebootmotion"


class MovementType(str, Enum):
    BASEBALL_PITCHING = "baseball-pitching"
    BASEBALL_HITTING = "baseball-hitting"
    BASKETBALL_SHOOTING = "basketball-shooting"
    VERTICAL_JUMPING = "vertical-jumping"
    SQUATTING = "squatting"
    LUNGING = "lunging"
    LEG_RAISING = "leg-raising"


MOVEMENT_TYPE_IDS_MAP = {
    MovementType.BASEBALL_HITTING.value: 1,
    MovementType.BASEBALL_PITCHING.value: 2,
    MovementType.BASKETBALL_SHOOTING.value: 67,
}


class Handedness(str, Enum):
    RIGHT = "right"
    LEFT = "left"
    NONE = "none"

    @property
    def eye_hand_multiplier(self) -> int:
        return 1 if self == Handedness.RIGHT else -1


class FileType(str, Enum):
    INVERSE_KINEMATICS = "inverse-kinematics"
    IK_AT_TIME_POINTS = "ik-at-time-points"
    MOMENTUM_ENERGY = "momentum-energy"
    HITTING_CALCULATED_SERIES = "hitting-processed-series"
    HITTING_CALCULATED_METRICS = "hitting-processed-metrics"
    HITTING_LITE_CALCULATED_SERIES = "hitting-lite-processed-series"
    HITTING_LITE_CALCULATED_METRICS = "hitting-lite-processed-metrics"
    METRICS_BASEBALL_PITCHING_V1 = "metrics-baseball-pitching-v1-0-0"
    METRICS_BASEBALL_PITCHING_V2 = "metrics-baseball-pitching-v2-0-0"
    METRICS_BASEBALL_PITCHING_V_ALL = "metrics-baseball-pitching"
    METRICS_BASEBALL_HITTING_V1 = "metrics-baseball-hitting-v1-0-0"
    METRICS_BASEBALL_HITTING_V2 = "metrics-baseball-hitting-v2-0-0"
    METRICS_BASEBALL_HITTING_V_ALL = "metrics-baseball-hitting"


HITTING_SERIES_TYPES = (
    FileType.HITTING_CALCULATED_SERIES.value,
    FileType.HITTING_LITE_CALCULATED_SERIES.value,
)


HITTING_METRICS_TYPES = (
    FileType.HITTING_CALCULATED_METRICS.value,
    FileType.HITTING_LITE_CALCULATED_METRICS.value,
)


class DataType(str, Enum):
    METADATA = "metadata"


@dataclass
class S3Metadata:
    org_id: str
    mocap_types: list[MocapType]
    movement_type: MovementType
    handedness: Handedness | None = None
    file_type: FileType | None = None

    @cached_property
    def bucket(self) -> str:
        s3_bucket = get_s3_bucket(org_id=self.org_id)
        if s3_bucket is None:
            raise RuntimeError(
                "s3 bucket must be set thru REBOOT_S3_BUCKET environment variable or by providing an org_id to S3Metadata"
            )
        return s3_bucket

    @property
    def mocap_type(self) -> str:
        assert len(self.mocap_types) != 0, "Must set some mocap types"
        if len(self.mocap_types) > 1:
            print(
                "Input multiple mocap_types, setting the mocap type as the first item in the list"
            )

        return self.mocap_types[0]

    @property
    def s3_population_prefix(self) -> str:
        return (
            f"s3://{self.bucket}/population/{self.mocap_type.value}/{self.movement_type.value}/"
            f"{self.file_type.value}/000000_{self.movement_type.value}_{self.handedness.value}_"
        )


@dataclass
class PlayerMetadata:
    org_player_ids: Optional[list[str]] = None
    session_dates: Optional[list[str]] = None
    session_nums: Optional[list[str]] = None
    session_date_start: Optional[str] = None
    session_date_end: Optional[str] = None
    year: Optional[int] = None
    org_movement_id: Optional[str] = None
    s3_metadata: Optional[S3Metadata] = None

    @property
    def s3_prefix(self) -> str:
        assert self.s3_metadata, "Must set s3 metadata before generating s3 prefix"

        if self.session_dates and self.session_nums and self.org_player_ids:
            if (
                len(self.session_dates) == 1
                and len(self.session_nums) == 1
                and len(self.org_player_ids) == 1
            ):
                if len(self.s3_metadata.mocap_types) > 1:
                    print(
                        "Input multiple mocap_types, setting the mocap_type as the first item in the list"
                    )

                mocap_type = self.s3_metadata.mocap_types[0]

                return (
                    f"s3://{self.s3_metadata.bucket}/data_delivery/{mocap_type.value}/"
                    f"{self.session_dates[0]}/{self.session_nums[0]}/{self.s3_metadata.movement_type.value}/"
                    f"{self.org_player_ids[0]}/{self.s3_metadata.file_type}/"
                )

        print("Unable to construct path with input parameters")


class Functions(str, Enum):
    BACKEND = "reboot_toolkit_backend"
    BACKEND_DEV = "reboot_toolkit_backend_dev"
    INVERSE_KINEMATICS = "reboot_toolkit_ik"


class InvocationTypes(str, Enum):
    SYNC = "RequestResponse"
    ASYNC = "Event"
