import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.utilities.parsing import AttributeDict

from deep_depth_transfer.data import KittiDataModuleFactory
from deep_depth_transfer.models.factory import ModelFactory
from deep_depth_transfer.models.utils import load_undeepvo_checkpoint
from deep_depth_transfer.utils import TensorBoardLogger, MLFlowLogger, LoggerCollection

parser = ArgumentParser(description="Run deep depth transfer on kitty")
parser.add_argument("--config", type=str, default="./config/model.yaml")
parser.add_argument("--frames", type=str, default=None)
parser.add_argument("--sequences", type=str, default="08")
parser.add_argument("--dataset", type=str, default="./datasets/kitti_odometry")
parser.add_argument("--load_model", type=bool, default=False)
parser.add_argument("--model_checkpoint", type=str, default="./checkpoints/checkpoint_undeepvo.pth")
parser.add_argument("--experiment_name", type=str, default="Pykitti")
parser = pl.Trainer.add_argparse_args(parser)
arguments = parser.parse_args()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://proxy2.cod.phystech.edu:10086/"
os.environ["AWS_ACCESS_KEY_ID"] = "depth"
os.environ["AWS_SECRET_ACCESS_KEY"] = "depth123"
mlflow_url = "http://proxy2.cod.phystech.edu:10085/"
logger = LoggerCollection(
    [TensorBoardLogger("lightning_logs"),
     MLFlowLogger(experiment_name=arguments.experiment_name, tracking_uri=mlflow_url)]
)

# Make trainer
trainer = pl.Trainer.from_argparse_args(arguments, logger=logger)

# Make data model factory
if arguments.frames is not None:
    frames = arguments.frames.split(",")
    frames = [int(x) for x in frames]
    frames = range(*frames)
else:
    frames = None
data_model_factory = KittiDataModuleFactory(frames, arguments.sequences, arguments.dataset)

# Load parameters
params = load_hparams_from_yaml(arguments.config)
params = AttributeDict(params)
params.frames = arguments.frames
print("Load model from params \n" + str(params))
data_model = data_model_factory.make_data_module_from_params(params)
model = ModelFactory().make_model(params, data_model.get_cameras_calibration())

if arguments.load_model:
    print("Load checkpoint")
    load_undeepvo_checkpoint(model, arguments.model_checkpoint)

print("Start training")
trainer.fit(model, data_model)
