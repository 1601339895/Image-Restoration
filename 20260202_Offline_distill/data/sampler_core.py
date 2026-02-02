# sampler_core.py
import os
from loguru import logger
import sampler_method as sampler
from frtrain.misc import utils


def build_sample_datasets(config):
    """
    等价于原 sampler.py 里：
    - load dataset_collection
    - load meta
    - 分配 start_label / mask
    """
    data_list_config = os.path.join(
        os.path.dirname(config["_config_file"]),
        config["datasets"]["dataset_collection"],
    )
    data_path = utils.load_config(data_list_config)

    datasets_cfg = config["datasets"]["datasets"]

    start_label = 0
    all_datasets = []

    for dataset in datasets_cfg:
        info_file = os.path.join(
            os.path.dirname(config["_config_file"]), dataset["path"]
        )
        data_info = utils.load_config(info_file)

        sample_datasets = []

        for d in data_info.get("datasets", []):
            name = d["name"]
            paths = data_path.get(name, [])

            meta = sampler.load_dataset(name, paths)
            if meta is None:
                continue

            d = d.copy()
            d["meta"] = meta
            d["sample_func"] = d.get("sample_method", "sampler_uniform_image")
            d["start_label"] = start_label
            d["num_class"] = len(meta["info"])
            d["mask"] = (start_label, start_label + d["num_class"])

            start_label += d["num_class"]
            sample_datasets.append(d)
            all_datasets.append(d)

        dataset["sample_datasets"] = sample_datasets

    logger.info(
        f"Sampler init done: total datasets={len(all_datasets)}, total classes={start_label}"
    )
    return datasets_cfg


def sample_one_batch(dataset_cfg, batch_size):
    """
    等价于 sampler.batch_sample
    """
    return sampler.batch_sample(dataset_cfg, batch_size)
