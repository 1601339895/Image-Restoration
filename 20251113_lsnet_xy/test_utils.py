import os
import sys
from loguru import logger
from nori2.utils import is_s3
import nori2 as nori


def load_nori(nori_id):
    str_b = fallback_nori_fetcher.get(nori_id)
    img_raw = cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_UNCHANGED)
    return img_raw

class FallbackNoriFetcher:
    """
    获取noriid
    """

    def __init__(self, fetcher, verbose=True):
        """
        :param fetcher:
        :param verbose: True
        """
        self.fetcher = fetcher
        self.verbose = verbose

    def get(self, data_id):
        """
        采用不同的方式获取nori_id
        :param data_id:
        :return: self.fetcher.get(data_id)
        """
        try:
            return self.fetcher.get(data_id, retry=0)
        except Exception as e:
            try:
                if self.verbose:
                    print(
                        'FUCK! Nori fetcher said failed to get {} because of {}. Fallback to read nori directly.'.format(
                            data_id, e), flush=True)
                v = int(data_id.split(',')[0])
                path = nori.speedup.locateNori(v)
                if is_s3(path):
                    pass
                else:
                    path = '/unsullied/sharefs/' + path
                if self.verbose:
                    print('Found volume {} at {}'.format(v, path), flush=True)

                with nori.open(path, 'r') as nr:
                    return nr.get(data_id)
            except Exception as e:
                print('FUCK! Failed to fallback get {} because of {}. Let\'s relay on the original fetcher.'.format(
                    data_id, e), flush=True)
                return self.fetcher.get(data_id)

fallback_nori_fetcher = FallbackNoriFetcher(nori.Fetcher(), verbose=False)


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")
