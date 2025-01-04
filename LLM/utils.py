import logging
import sys
from typing import Tuple
from transformers import AutoModelForCausalLM, LlamaTokenizer, PreTrainedModel, PreTrainedTokenizer
from config import *
import torch
import tqdm

class LoggerWriter:
    def __init__(self, logfct):
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith('\n'):
            self.buf.append(msg.removesuffix('\n'))
            self.logfct(''.join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass


class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(logging_path: str,mode='w'):
    if not logging_path.endswith('.log'):
        logging_path += '.log'
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logging_path, mode=mode)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(ColorFormatter())
    logger.addHandler(console)
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)
    return logger


def set_gpu(gpus: list, logger):
    if type(gpus) is int:
        gpus = [gpus]
    gpus = ','.join([str(i) for i in gpus])
    logger.debug(f'using GPU [{gpus}]')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus


def load_model_and_tokenizer(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch.bfloat16,
                                                 proxies=proxies, use_auth_token=hf_token, cache_dir=cache_dir,
                                                 device_map='auto')

    tokenizer = LlamaTokenizer.from_pretrained(model_name, proxies=proxies, use_auth_token=hf_token,
                                               cache_dir=cache_dir, padding_side='left')

    # add pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
