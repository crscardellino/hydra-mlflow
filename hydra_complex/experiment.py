#!/usr/bin/env python

import hydra
import logging
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver('eval', lambda x: eval(x))
    logger.info(f"Loading dataset from {cfg.input.data_file}")
    data = pd.read_csv(cfg.input.data_file)

    train_data = data.loc[data['Split'] == cfg.train.split].iloc[:, 2:].values
    train_target = data.loc[data['Split'] == cfg.train.split, 'Quality'].values

    eval_data = data.loc[data['Split'] == cfg.evaluation.split].iloc[:, 2:].values
    eval_target = data.loc[data['Split'] == cfg.evaluation.split, 'Quality'].values

    logger.info("Training classification model")
    clf = cfg.train.model.module(**cfg.train.model.params).fit(train_data, train_target)

    eval_preds = clf.predict(eval_data)
    logger.info(f"Classification results for {cfg.evaluation.split} split:\n" +
                classification_report(eval_target, eval_preds))

    logger.info("The model was trained with the following parameters:\n" +
                OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
