import argparse
import os
import sys
import time

import numpy as np
import yaml
from dpflow import InputPipe, OutputPipe, control
from loguru import logger
import sys

from frtrain.misc import utils

logger.remove()
logger.add(sys.stderr, colorize=True)


def data_transfer(qi, new_batch_size, datasets):
    data = qi.get()
    data_new = []
    data_tmp = {}

    for dataset in datasets:
        data_tmp = {}
        key_list = list(data[dataset["name"]].keys())
        for key in key_list:
            data_tmp[key] = np.copy(data[dataset["name"]][key])
            data_tmp[key] = data_tmp[key].reshape(
                -1, new_batch_size * dataset["imgs"], *data_tmp[key].shape[1:]
            )
        batch_split = int(
            data[dataset["name"]][key_list[0]].shape[0]
            / (new_batch_size * dataset["imgs"])
        )
        for i in range(batch_split):
            if len(data_new) <= i:
                data_new.append({})
            for key in key_list:
                if key != 'tea_nr_keys':
                    data_new[i]["{}:{}".format(key, dataset["name"])] = data_tmp[key][i]
    return data_new


class new_control:
    def __init__(self, qos):
        self.control_list = [control(io=[qi])] + [control(io=[i]) for i in qos]

    def __enter__(self):
        for i in self.control_list:
            i.__enter__()

    def __exit__(self):
        for i in self.control_list:
            i.__exit__()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dp_split")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "config.yaml"),
        help="train config",
    )

    args = parser.parse_args()
    config = utils.load_config(args.config)

    # calculate img num
    img_num = 0
    for dataset in config["datasets"]["datasets"]:
        img_num += dataset["imgs"]

    input_pipe = config["provider_pipe_name"]
    output_pipe = "{}.{}".format(input_pipe, os.getcwd().split("/")[-1])
    nodes = config["nodes"]
    gpu_per_mathine = config["gpu_per_machine"]
    world_size = nodes * gpu_per_mathine
    batch_size = config["batch_size"]
    ddp_real = config["ddp_real"]
    new_batch_size = batch_size // world_size
    if not ddp_real:
        new_batch_size = new_batch_size * world_size

    qos = []
    for i in range(world_size):
        logger.info(output_pipe + "." + str(i))
        qos.append(OutputPipe(output_pipe + "." + str(i), buffer_size=10))
    qi = InputPipe(input_pipe)
    qi._meta = {"group_id": output_pipe}

    t = []
    rank_index = 0
    with new_control(qos):
        while True:
            t.append([time.time()])
            data = data_transfer(qi, new_batch_size, config["datasets"]["datasets"])
            for batch_data in data:
                if ddp_real:
                    if not qos[rank_index].full():
                        qos[rank_index].put_pyobj(batch_data)
                    rank_index = (rank_index + 1) % world_size
                else:
                    for qos_tmp in qos:
                        if not qos_tmp.full():
                            qos_tmp.put_pyobj(batch_data)
            t[-1].append(time.time())
            t = t[-1000:]
            logger.info(
                "get data: {:4.4f} | get data average: {:4.4f}".format(
                    t[-1][-1] - t[-1][-2], (t[-1][-1] - t[0][0]) / len(t)
                )
            )
