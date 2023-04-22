from typing import Optional
from enum import Enum
from dataclasses import dataclass


class MovementType(str, Enum):
    BASEBALL_PITCHING = "baseball-pitching"
    BASEBALL_HITTING = "baseball-hitting"


class Handedness(str, Enum):
    RIGHT = "right"
    LEFT = "left"

    @property
    def eye_hand_multiplier(self) -> int:
        return 1 if self == Handedness.RIGHT else -1


@dataclass
class S3Metadata:
    org_id: str
    mocap_type: str
    movement_type: MovementType
    handedness: Handedness
    file_type: str

    @property
    def s3_population_prefix(self) -> str:
        return f"s3://reboot-motion-{self.org_id}/population/{self.mocap_type}/{self.movement_type}/" \
               f"{self.file_type}/000000_{self.movement_type}_{self.handedness}_"


@dataclass
class PlayerMetadata:
    mlbam_player_ids: Optional[list[str]] = None
    session_dates: Optional[list[str]] = None
    game_pks: Optional[list[str]] = None
    session_date_start: Optional[str] = None
    session_date_end: Optional[str] = None
    year: Optional[int] = None
    mlb_play_guid: Optional[str] = None
    s3_metadata: Optional[S3Metadata] = None

    @property
    def s3_prefix(self) -> str:
        assert self.s3_metadata, "Must set s3 metadata before generating s3 prefix"
        if self.session_dates and self.game_pks and self.mlbam_player_ids:
            if len(self.session_dates) == 1 and len(self.game_pks) == 1 and len(self.mlbam_player_ids) == 1:
                return f's3://reboot-motion-{self.s3_metadata.org_id}/data_delivery/{self.s3_metadata.mocap_type}/' \
                       f'{self.session_dates[0]}/{self.game_pks[0]}/{self.s3_metadata.movement_type}/' \
                       f'{self.mlbam_player_ids[0]}/{self.s3_metadata.file_type}/'


class Functions(str, Enum):
    VISUALIZATIONS = "colab_notebook_backend"
    INVERSE_KINEMATICS = "toolbox_ik"


class InvocationTypes(str, Enum):
    SYNC = "RequestResponse"
    ASYNC = "Event"
