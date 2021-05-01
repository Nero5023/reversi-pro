from multiprocessing import Pool, Process
from nnet import NeuralNet
from selfplay import SelfPlayBatch
from config import game_config
import config
import os
from multiprocessing import Pool, Process
import json
from util import flatten
import torch
import argparse

BEST_CHECKPOINT_FN = "best_model.tar"
BEST_MODEL_TYPE = 1

CHECK_POINT_FN_TEM = 'model_v{}.tar'


def load_train_status():
    file_path = config.train_status_path
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = config.train_status_default.copy()
    return data


def save_train_status(data):
    file_path = config.train_status_path
    with open(file_path, 'w') as f:
        json.dump(data, f)


def get_checkpoint_folder(type=1):
    if type == 1:
        return 'checkpoint'
    else:
        return 'checkpoint_v2'


def load_model_with_version(version, type=1):
    nn = NeuralNet(game_config, model_type=type)
    if version is None:
        return nn
    fn = CHECK_POINT_FN_TEM.format(version)
    checkpoint_dir = get_checkpoint_folder(type)

    f_path = checkpoint_dir + '/' + fn
    if os.path.isfile(f_path):
        nn.load_checkpoint(folder=checkpoint_dir, filename=fn)
        return nn
    raise Exception("Model {} not found.".format(f_path))


def delete_model(current_version, type=1):
    if current_version < 5:
        return
    if current_version % 10 == 0:
        return
    delete_version = current_version - 10
    fn = CHECK_POINT_FN_TEM.format(delete_version)
    folder = get_checkpoint_folder(type)
    f_path = folder + '/' + fn
    if os.path.isfile(f_path):
        os.remove(f_path)


class TrainPipe:
    def __init__(self, parallel_num=config.self_play_parallel_num):
        self.train_status = load_train_status()
        self.parallel_num = parallel_num

    @property
    def version(self):
        return self.train_status.get('version', None)

    @version.setter
    def version(self, value):
        self.train_status['version'] = value

    def start(self):
        while True:
            pool = Pool(config.self_play_parallel_num)
            # for i in range(config.self_play_parallel_num):
            #     pool.apply_async(self.self_play_game, (i, self.version))
            datas = pool.map(self_play_game_worker, [(i, self.version) for i in range(self.parallel_num)])
            pool.close()
            pool.join()

            datas = flatten(datas)

            process = Process(target=train_worker, args=(datas, self.version))
            process.start()
            process.join()
            if self.version is None:
                self.version = 0
            else:
                self.version = self.version + 1
            save_train_status(self.train_status)
            delete_model(self.version)


def self_play_game_worker(arg):
    i, version = arg
    if version is None or version <= 2:
        nn = NeuralNet(game_config, model_type=BEST_MODEL_TYPE)
        fdir = get_checkpoint_folder(BEST_MODEL_TYPE)
        if os.path.isfile(fdir + '/' + BEST_CHECKPOINT_FN):
            nn.load_checkpoint(folder=fdir, filename=BEST_CHECKPOINT_FN)
            print("Playing best model")
        else:
            print("Playing with non model")
    else:
        nn = load_model_with_version(version, config.train_model_type)
        print("playing: v{} type:{}".format(version, config.train_model_type))
    game = SelfPlayBatch(nn)
    game.start()
    return game.game_data


def train_worker(data, version):
    print("training: v{} type:{}".format(version, config.train_model_type))
    nn = load_model_with_version(version, config.train_model_type)
    nn.train(data, version)
    new_version = 0
    if version is not None:
        new_version = version + 1
    nn.save_checkpoint(filename=CHECK_POINT_FN_TEM.format(new_version))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--type', type=int, dest="type", default=1, required=False)
    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config.train_model_type = args.type
    print("training type:{}".format(config.train_model_type))
    if torch.cuda.is_available():
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    train = TrainPipe()
    train.start()
