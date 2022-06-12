# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf

import optuna

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    
    def objective(trial):
        FLAGS.config.model.num_scales = trial.suggest_int("num_scales", 900, 1200, step=100)
        FLAGS.config.model.beta_max = trial.suggest_discrete_uniform("beta_max", 10, 30, 10)
        FLAGS.config.model.nonlinearity = trial.suggest_categorical("nonlinearity", ["swish", "relu"])
        FLAGS.config.optim.lr = trial.suggest_float("lr", 1e-4, 4e-4, step=1e-4)
        FLAGS.config.model.discount_sigma = trial.suggest_float("discount_sigma", 0.7, 1.2, step=0.1)

        # Create the working directory
        tf.io.gfile.makedirs(FLAGS.workdir)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        
#       if FLAGS.mode == "train":
        # Run the training pipeline
        run_lib.train(FLAGS.config, FLAGS.workdir)
#       elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        bpd, fid = run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
        return bpd + fid / 30.0
        
    study = optuna.create_study()
#     create_study(direction = "maximize")
    study.optimize(objective, n_trials=80)
  
    torch.save(study, 'study.pth')

    study.best_params  # E.g. {'x': 2.002108042}
    trial = study.best_trial
    print("Best Score: ", trial.value)
    print("Best Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))
    optuna.visualization.plot_contour(study)

if __name__ == "__main__":
  app.run(main)
