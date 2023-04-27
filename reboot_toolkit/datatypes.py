from typing import Optional
from enum import Enum
from dataclasses import dataclass


class MocapType(str, Enum):
    HAWKEYE = "hawkeye"
    HAWKEYE_HFR = "hawkeyehfr"


class MovementType(str, Enum):
    BASEBALL_PITCHING = "baseball-pitching"
    BASEBALL_HITTING = "baseball-hitting"


class Handedness(str, Enum):
    RIGHT = "right"
    LEFT = "left"

    @property
    def eye_hand_multiplier(self) -> int:
        return 1 if self == Handedness.RIGHT else -1


class FileType(str, Enum):
    INVERSE_KINEMATICS = "inverse-kinematics"
    IK_AT_TIME_POINTS = "ik-at-time-points"
    METRICS_BASEBALL_PITCHING_V1 = "metrics-baseball-pitching-v1-0-0"
    METRICS_BASEBALL_PITCHING_V2 = "metrics-baseball-pitching-v2-0-0"
    METRICS_BASEBALL_HITTING_V1 = "metrics-baseball-hitting-v1-0-0"
    METRICS_BASEBALL_HITTING_V2 = "metrics-baseball-hitting-v2-0-0"


@dataclass
class S3Metadata:
    org_id: str
    mocap_types: list[MocapType]
    movement_type: MovementType
    handedness: Handedness
    file_type: FileType

    @property
    def s3_population_prefix(self) -> str:
        if len(self.mocap_types) > 1:
            print('Input multiple mocap_types, setting the mocap type as the first item in the list')

        mocap_type = self.mocap_types[0]

        return f"s3://reboot-motion-{self.org_id}/population/{mocap_type}/{self.movement_type}/" \
               f"{self.file_type}/000000_{self.movement_type}_{self.handedness}_"


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
            if len(self.session_dates) == 1 and len(self.session_nums) == 1 and len(self.org_player_ids) == 1:
                if len(self.s3_metadata.mocap_types) > 1:
                    print('Input multiple mocap_types, setting the mocap_type as the first item in the list')

                mocap_type = self.s3_metadata.mocap_types[0]
                return f's3://reboot-motion-{self.s3_metadata.org_id}/data_delivery/{mocap_type}/' \
                       f'{self.session_dates[0]}/{self.session_nums[0]}/{self.s3_metadata.movement_type}/' \
                       f'{self.org_player_ids[0]}/{self.s3_metadata.file_type}/'

        print("Unable to construct path with input parameters")


class Functions(str, Enum):
    VISUALIZATIONS = "reboot_toolkit_backend"
    INVERSE_KINEMATICS = "reboot_toolkit_ik"


class InvocationTypes(str, Enum):
    SYNC = "RequestResponse"
    ASYNC = "Event"