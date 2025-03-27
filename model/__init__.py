import sys

sys.path.append('model')
from .poseCnn import PoseKeypointCNN
from .poseDct import PoseKeypointDCT
from .poseGCN import PoseKeypointGAT, create_complete_graph
from .poseGCN import PoseKeypointGAT_residual