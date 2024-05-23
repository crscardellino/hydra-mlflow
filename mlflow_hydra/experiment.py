#!/usr/bin/env python
"""
Main module for the MLFlow Hydra Experimentation Framework.

    MLFlow Hydra Experimentation Framework
    Copyright (C) 2024 Cristian Cardellino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import hydra
import logging
import mlflow
import numpy as np
import pandas as pd
import torch

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from tempfile import TemporaryDirectory
from torch.utils.data import TensorDataset, DataLoader

from .utils import flatten_dict
from .model import MultiLayerPerceptron

logger = logging.getLogger(__name__)


def run_experiment(cfg: DictConfig, run_id: str):
    """
    Main function in charge of running the experiment.

    Parameters
    ----------
        cfg: DictConfig
            Configuration dictionary for the run given by Hydra.
        run_id: str
            Run id of the current MLFlow run.
    """
    seed_everything(cfg.train.random_seed)
    logger.info(f"Loading dataset from {cfg.input.data_file}")
    data = pd.read_csv(cfg.input.data_file)

    train_data = data.loc[data['Split'] == 'train'].iloc[:, 2:].values
    train_target = data.loc[data['Split'] == 'train', 'Quality'].values
    train_target = train_target - 1  # Reduce the value to the [0,2] interval to simplify

    val_data = data.loc[data['Split'] == 'validation'].iloc[:, 2:].values
    val_target = data.loc[data['Split'] == 'validation', 'Quality'].values
    val_target = val_target - 1  # Reduce the value to the [0,2] interval to simplify

    if cfg.train.feature_scaling:
        logger.info("Scaling features")
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        val_data = scaler.transform(val_data)

    train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_target))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )

    val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.LongTensor(val_target))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    early_stopping = EarlyStopping(
        monitor='mlp__validation_loss',
        min_delta=1e-5,
        patience=cfg.train.early_stop
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.input.experiment_name,
        run_name=cfg.input.run_name,
        run_id=run_id
    )

    logger.info("Building and training classification model")
    model = MultiLayerPerceptron(
        input_size=train_data.shape[1],
        output_size=len(set(train_target)),
        **cfg.train.model
    )

    trainer = Trainer(
        logger=mlflow_logger,
        max_epochs=cfg.train.epochs,
        callbacks=[early_stopping],
        log_every_n_steps=max(len(train_dataloader) // 100, 1)
    )

    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    logger.info("Finished training classification model")

    if cfg.train.test_evaluation:
        logger.info("Evaluating final model using test split")
        eval_data = data.loc[data['Split'] == 'test'].iloc[:, 2:].values
        eval_target = data.loc[data['Split'] == 'test', 'Quality'].values
        eval_target = eval_target - 1
        if cfg.train.feature_scaling:
            eval_data = scaler.transform(eval_data)
    else:
        logger.info("Evaluating final model using validation split")
        eval_data = val_data
        eval_target = val_target

    eval_dataset = TensorDataset(torch.FloatTensor(eval_data))
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    eval_probs = np.concatenate([
        pred.detach().cpu().numpy() for pred in trainer.predict(model, eval_dataloader)
    ])
    eval_preds = eval_probs.argmax(axis=1)
    accuracy = accuracy_score(eval_target, eval_preds)
    f1 = f1_score(eval_target, eval_preds, average="macro")
    report = f"**Classification results**\n```{classification_report(eval_target, eval_preds)}```"

    logger.info("Logging evaluation results on mlflow")
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1
    })
    mlflow.set_tag("mlflow.note.content", report)

    logger.info("Logging evaluation predictions as artifacts")
    predictions_features = pd.DataFrame(eval_data, columns=data.columns[2:])
    target_predictions = pd.Series(eval_target, name="Quality")
    predictions_probs = pd.DataFrame(
        eval_probs,
        columns=[f"Quality {i+1} Prediction Probability" for i in range(len(set(eval_target)))]
    )
    predictions_dataset = pd.concat(
        [target_predictions, predictions_probs, predictions_features],
        axis=1
    )
    with TemporaryDirectory() as tmpdir:
        # We create a temporal directory to locally store some of the artifacts
        # before logging them
        predictions_path = Path(tmpdir) / 'predictions.csv'
        predictions_dataset.to_csv(predictions_path, index=False)
        mlflow.log_artifact(predictions_path)


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    """
    Main Hydra application that runs the experiment.

    Parameters
    ----------
        cfg: DictConfig
            The Hydra configuration dictionary.
    """
    OmegaConf.register_new_resolver('eval', lambda x: eval(x))

    mlflow.set_experiment(cfg.input.experiment_name)
    mlflow.set_experiment_tag('mlflow.note.content', cfg.input.experiment_description)

    with mlflow.start_run(run_name=cfg.input.run_name) as run:
        logger.info("Logging configuration as artifact")
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            with open(config_path, "wt") as fh:
                print(OmegaConf.to_yaml(cfg, resolve=False), file=fh)
            mlflow.log_artifact(config_path)

        logger.info("Logging train parameters")
        # Log params expects a flatten dictionary, since the configuration has nested
        # configurations (e.g. train.model), we need to use flatten_dict in order to
        # transform it into something that can be easily logged by MLFlow
        mlflow.log_params(flatten_dict(OmegaConf.to_container(cfg, resolve=False)))
        run_experiment(cfg, run.info.run_id)


if __name__ == "__main__":
    main()
