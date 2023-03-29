from enum import Enum


class Functions(str, Enum):
    VISUALIZATIONS = "colab_notebook_backend"
    INVERSE_KINEMATICS = "toolbox_ik"


class InvocationTypes(str, Enum):
    SYNC = "RequestResponse"
    ASYNC = "Event"
