import pickle
import numpy as np
import refile
import nori2 as nori
from loguru import logger
from frtrain.data.nori_compress import CompactNoriIDSet

nf = nori.Fetcher()

def load_dataset(name, paths):
    def _load(path, key):
        try:
            with refile.smart_open(refile.smart_path_join(path, key), "rb") as f:
                return np.array(pickle.load(f))
        except Exception as e:
            logger.error(f"Failed to load {key} from {path}: {e}")
            raise

    def _remove_waste_img(info, nr_keys):
        try:
            logger.info(f"Processing waste images: info.shape={info.shape}")
            
            if len(info.shape) == 3:
                logger.info("Using NEW data format processing (3D info)")
                sort_index = np.argsort(info, axis=0)[:, 0, 0]
                info_reshape = info[sort_index].reshape(-1, 2)
            else:
                logger.info("Using OLD data format processing (2D info)")
                sort_index = np.argsort(info, axis=0)[:, 0]
                info_reshape = info[sort_index]
                
            diff = info_reshape[:, 0] - (np.concatenate([np.array([0]), info_reshape[:-1, 1]]))
            diff_sum = np.cumsum(diff)
            diff_final = np.repeat(diff_sum, 2).reshape(-1, 2)
            info_res = info_reshape - diff_final
            
            if len(info.shape) == 3:
                info_res = info_res.reshape(-1, 2, 2)
                
            waste_img = []
            for end, num in zip(info_reshape[:, 0][diff != 0], diff[diff != 0]):
                waste_img += list(range(end - num, end))
                
            if len(waste_img) > 0:
                logger.info(f"Removing {len(waste_img)} waste images")
                
            nr_keys = np.delete(nr_keys, waste_img)
            logger.info(f"After cleanup: {len(info_res)} persons, {len(nr_keys)} images")
            
            return info_res, nr_keys
        except Exception as e:
            logger.error(f"Error in _remove_waste_img: {e}")
            raise

    info = []
    nr_keys = CompactNoriIDSet()
    valid = False

    for path in paths:
        try:
            logger.info(f"Loading path {path} for dataset {name}")
            info_tmp = _load(path, "info")
            nr_keys_tmp = _load(path, "new_align5p.nori_id")
            
            logger.info(f"Raw data: info_shape={info_tmp.shape}, nr_keys_type={type(nr_keys_tmp)}")
            
            info_tmp, nr_keys_tmp = _remove_waste_img(info_tmp, nr_keys_tmp)
            
            if len(info) > 0:
                offset = info[-1].max() if info else 0
                logger.info(f"Applying offset {offset} to align with existing data")
                info_tmp = info_tmp + offset
                
            info.append(info_tmp)
            nr_keys.extend(nr_keys_tmp)
            valid = True
            
            logger.info(f"Path {path}: +{len(info_tmp)} persons, +{len(nr_keys_tmp)} images")
            
        except Exception as e:
            logger.warning(f"Skipping path {path} for dataset {name} due to error: {e}")
            continue

    if not valid:
        logger.error(f"No valid paths loaded for dataset {name}")
        return None

    try:
        info = np.concatenate(info)
        labels = np.array([l for l, idx in enumerate(info) for _ in range(idx.min(), idx.max())])
        meta = {"name": name, "info": info, "nr_keys": nr_keys, "labels": labels}
        
        logger.info(f"Final dataset {name}: #persons={len(info)}, #images={len(nr_keys)}")
        logger.info(f"Labels range: [{labels.min()}-{labels.max()}]")
        
        return meta
    except Exception as e:
        logger.error(f"Error constructing meta for dataset {name}: {e}")
        return None


def sampler_uniform_image(meta, img_num):
    if meta is None or len(meta["nr_keys"]) == 0:
        return []
    return np.random.randint(0, len(meta["nr_keys"]), img_num)


def sampler_uniform_person_bq(meta, img_num, balance_base_query=True):
    if meta is None or len(meta["info"]) == 0:
        return []
        
    def _sampling_info(index_list, img_num):
        if len(index_list) >= img_num:
            return list(np.random.choice(index_list, img_num, replace=False))
        else:
            return list(np.random.choice(index_list, img_num, replace=True))

    try:
        i = np.random.randint(0, len(meta["info"]))
        selected_person_info = meta["info"][i]
        
        # 采样过程打印
        if np.random.random() < 0.01:  # 1%概率打印
            logger.info(f"Sampling person {i}: info={selected_person_info}")
        
        if len(selected_person_info.shape) == 1:
            result = _sampling_info(range(*selected_person_info), img_num)
            if np.random.random() < 0.01:
                logger.info(f"1D format: sampled indices {result}")
            return result

        (sb, eb), (sq, eq) = selected_person_info
        assert sb <= eb == sq <= eq
        
        if sb == eb or sq == eq:
            result = _sampling_info(range(selected_person_info.min(), selected_person_info.max()), img_num)
            if np.random.random() < 0.01:
                logger.info(f"Single range: sampled indices {result}")
            return result

        if not balance_base_query:
            all_range = list(range(sb, eb)) + list(range(sq, eq))
            result = _sampling_info(all_range, img_num)
            if np.random.random() < 0.01:
                logger.info(f"All range: sampled {len(result)} from base[{sb}-{eb}] + query[{sq}-{eq}]")
            return result

        if img_num == 1:
            if np.random.randn() > 0:
                result = _sampling_info(range(sb, eb), 1)
                source = "base"
            else:
                result = _sampling_info(range(sq, eq), 1)
                source = "query"
            if np.random.random() < 0.01:
                logger.info(f"Single sample from {source}: {result}")
            return result
        else:
            base_samples = _sampling_info(range(sb, eb), 1)
            query_samples = _sampling_info(range(sq, eq), img_num - 1)
            result = base_samples + query_samples
            if np.random.random() < 0.01:
                logger.info(f"Balanced sampling: 1 from base + {img_num-1} from query = {result}")
            return result
            
    except Exception as e:
        logger.error(f"Error in sampler_uniform_person_bq: {e}")
        return []


def sampler_uniform_person_qq(meta, img_num, balance_base_query=True):
    if meta is None or len(meta["info"]) == 0:
        return []
    def _sampling_info(index_list, img_num):
        if len(index_list) >= img_num:
            return list(np.random.choice(index_list, img_num, replace=False))
        else:
            return list(np.random.choice(index_list, img_num, replace=True))

    try:
        i = np.random.randint(0, len(meta["info"]))
        if len(meta["info"][i].shape) == 1:
            return _sampling_info(range(*meta["info"][i]), img_num)

        (sb, eb), (sq, eq) = meta["info"][i]
        assert sb <= eb == sq <= eq
        if sb == eb or sq == eq:
            return _sampling_info(range(meta["info"][i].min(), meta["info"][i].max()), img_num)

        if not balance_base_query:
            return _sampling_info(list(range(sb, eb)) + list(range(sq, eq)), img_num)

        if img_num == 1:
            if np.random.randn() > 0:
                return _sampling_info(range(sb, eb), 1)
            else:
                return _sampling_info(range(sq, eq), 1)
        else:
            return _sampling_info(range(sq, eq), 1) + _sampling_info(range(sq, eq), img_num - 1)
    except Exception as e:
        logger.error(f"Error in sampler_uniform_person_qq: {e}")
        return []


def sampler_uniform_person_bb(meta, img_num, balance_base_query=True):
    if meta is None or len(meta["info"]) == 0:
        return []
    def _sampling_info(index_list, img_num):
        if len(index_list) >= img_num:
            return list(np.random.choice(index_list, img_num, replace=False))
        else:
            return list(np.random.choice(index_list, img_num, replace=True))

    try:
        i = np.random.randint(0, len(meta["info"]))
        if len(meta["info"][i].shape) == 1:
            return _sampling_info(range(*meta["info"][i]), img_num)

        (sb, eb), (sq, eq) = meta["info"][i]
        assert sb <= eb == sq <= eq
        if sb == eb or sq == eq:
            return _sampling_info(range(meta["info"][i].min(), meta["info"][i].max()), img_num)

        if not balance_base_query:
            return _sampling_info(list(range(sb, eb)) + list(range(sq, eq)), img_num)

        if img_num == 1:
            if np.random.randn() > 0:
                return _sampling_info(range(sb, eb), 1)
            else:
                return _sampling_info(range(sq, eq), 1)
        else:
            return _sampling_info(range(sb, eb), 1) + _sampling_info(range(sb, eb), img_num - 1)
    except Exception as e:
        logger.error(f"Error in sampler_uniform_person_bb: {e}")
        return []


def sample_dataset(datasets, batch_size):
    valid_datasets = [d for d in datasets if d.get("meta") is not None]
    if not valid_datasets:
        return []
    weights = np.array([d["weight"] for d in valid_datasets])
    weights = weights / weights.sum()
    sampled_indices = np.random.choice(len(valid_datasets), batch_size, p=weights)
    return sampled_indices


def batch_sample(datasets, batch_size):
    nr_keys, labels, masks, dataset_names = [], [], [], []  # 
    img_num = datasets.get("imgs", 0)
    sample_indices = sample_dataset(datasets.get("sample_datasets", []), batch_size)
    
    # 采样统计
    dataset_counts = {}
    
    for idx in sample_indices:
        try:
            """
            dset.keys():dict_keys(['name', 'sample_method', 'weight', 'meta', 'sample_func', 'start_label', 'num_class', 'mask'])
            """    
            dset = datasets["sample_datasets"][idx]
            dataset_name = dset.get("name", f"dataset_{idx}")   # 
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
            
            meta = dset.get("meta")
            if meta is None:
                continue
                
            sample_func = globals().get(dset["sample_func"])
            if not callable(sample_func):
                logger.error(f"Sample function {dset['sample_func']} not found")
                continue
                
            indices = sample_func(meta, img_num)
            for i in indices:
                nr_keys.append(meta["nr_keys"][i])
                labels.append(meta["labels"][i] + dset["start_label"])
                masks.append(dset["mask"])
                dataset_names.append(dataset_name)  # <-- 
        except Exception as e:
            logger.error(f"Error processing dataset index {idx}: {e}")
            continue
    
    # batch统计打印
    if np.random.random() < 0.05:  # 5%概率打印
        logger.info(f"Batch sample summary: {dataset_counts}")
        
    return {
        "nr_keys": np.array(nr_keys),
        "labels": np.array(labels),
        "masks": np.array(masks),
        "dataset_names": dataset_names,   #每个数据集的名称
    }
