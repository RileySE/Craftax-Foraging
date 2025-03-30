import csv
import datetime
from collections import defaultdict

import wandb
from termcolor import colored
import chex
from absl import logging
from pathlib import Path

COMMON_TRAIN_FORMAT = [
    ("steps", "S", "int"),
    ("fps", "FPS", "float"),
    ("episode_length", "AVG_LEN", "int"),
    ("episode_return", "AVG_R", "float"),
    ("episodes_done", "AVG_E", "int"),
    ("total_episodes", "TOTAL_EPISODES", "int"),
    ("total_returns", "TOTAL_RETURNS", "float"),
    ("task", "TASK", "int"),
    ("exploration_epsilon", "EPS", "float"),
    ("total_time", "T", "time"),
    ("score", "SCORE", "float"),
]


COMMON_EVAL_FORMAT = [
    ("steps", "S", "int"),
    ("fps", "FPS", "float"),
    ("episode_length", "AVG_LEN", "int"),
    ("episode_return", "AVG_R", "float"),
    ("episodes_done", "AVG_E", "int"),
    ("total_episodes", "TOTAL_EPISODES", "int"),
    ("total_returns", "TOTAL_RETURNS", "float"),
    ("task", "TASK", "int"),
    ("total_time", "T", "time"),
    ("score", "SCORE", "float"),
]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formatting, use_wandb):
        self._csv_file_name = csv_file_name
        self._formatting = formatting
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None
        self.use_wandb = use_wandb

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith("train"):
                key = key[len("train") + 1 :]
            elif key.startswith("eval"):
                key = key[len("eval") + 1 :]

            key = key.replace("/", "_")
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "num_episodes" not in row:
                    continue
                if float(row["num_episodes"]) >= data["num_episodes"]:
                    break
                rows.append(row)
        with self._csv_file_name.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(data.keys()), restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open("a")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=sorted(data.keys()),
                restval=0.0,
                extrasaction="ignore",
            )
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == "int":
            value = int(value)
            return f"{key}: {value}"
        elif ty == "float":
            return f"{key}: {value:.04f}"
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f"{key}: {value}"
        else:
            raise f"invalid format type: {ty}"

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow" if prefix == "train" else "green")
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formatting:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        logging.info(" | ".join(pieces))

    def _dump_to_wandb(self, data):
        wandb.log(data)

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["frame"] = step
        if self.use_wandb:
            wandb_data = {prefix + "/" + key: val for key, val in data.items()}
            self._dump_to_wandb(data=wandb_data)
        # self._dump_to_csv(data)   # comment out since I am not using csv
        self._dump_to_console(data, prefix)
        self._meters.clear()

    def dump_to_console(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["frame"] = step
        self._dump_to_console(data, prefix)

    def dump_to_wandb(self, step, prefix):
        if self.use_wandb:
            if len(self._meters) == 0:
                return
            data = self._prime_meters()
            data["frame"] = step
            if self.use_wandb:
                wandb_data = {prefix + "/" + key: val for key, val in data.items()}
                self._dump_to_wandb(data=wandb_data)

    def clear(self):
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_wandb):
        self._log_dir = log_dir
        # check if log_dir is path, if it is not, then convert it to path
        if not isinstance(log_dir, Path):
            log_dir_path = Path(log_dir)
        else:
            log_dir_path = log_dir
        self._train_mg = MetersGroup(
            log_dir_path / "train.csv", formatting=COMMON_TRAIN_FORMAT, use_wandb=use_wandb
        )
        self._eval_mg = MetersGroup(
            log_dir_path / "eval.csv", formatting=COMMON_EVAL_FORMAT, use_wandb=use_wandb
        )

        self._iteration_mg = MetersGroup(
            log_dir_path / "eval.csv", formatting=COMMON_EVAL_FORMAT, use_wandb=use_wandb
        )

        self._sw = None
        self.use_wandb = use_wandb

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        assert (
            key.startswith("train") or key.startswith("eval_") or key.startswith("eval")
        )
        chex.assert_rank([value], [0])
        self._try_sw_log(key, value, step)
        if key.startswith("eval"):
            mg = self._eval_mg
        else:
            mg = self._train_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f"{ty}/{key}", np.asarray(value), step)

    def dump(self, step, ty=None):
        if ty is None or ty == "eval":
            self._eval_mg.dump(step, "eval")
        if ty is None or ty == "train":
            self._train_mg.dump(step, "train")

    def dump_to_console(self, step, ty=None):
        if ty is None or ty == "eval":
            self._eval_mg.dump_to_console(step, "eval")
        if ty is None or ty == "train":
            self._train_mg.dump_to_console(step, "train")

    def dump_to_wandb(self, step, ty=None):
        if ty is None or ty == "eval":
            self._eval_mg.dump_to_wandb(step, "eval")
        if ty is None or ty == "train":
            self._train_mg.dump_to_wandb(step, "train")

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)

    def clear(self, ty):
        if ty is None or ty == "eval":
            self._eval_mg.clear()
        if ty is None or ty == "train":
            self._train_mg.clear()


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f"{self._ty}/{key}", value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)
