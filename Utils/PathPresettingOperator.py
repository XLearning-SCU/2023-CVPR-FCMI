import os
import torch

Server3090 = True
TiTan1 = True


class PathOperator:
    def __init__(self, model_name=None, checkpoint_storage_name=None, code_run_set_name=None):
        """
        'Codes',                   checkpoint_storage_name, self.code_run_set_name, code_fold_name
        'Checkpoints', model_name, checkpoint_storage_name,                         code_fold_name, checkpoint_epoch
        """
        current_gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        tensorboard_arg = ''
        if 'NVIDIA GeForce GTX 1050' == current_gpu:
            home = "D:/L/Deal/"
            __python_path = 'python'
            self.cuda_num = 1
        elif 'TITAN RTX' == current_gpu:
            if TiTan1:
                home = "/home/pengxin/"
                __python_path = 'anaconda3/envs/torch/bin/python3.6'
                self.ip = "192.168.50.31"
            else:
                home = "/deva/pengxin/"
                __python_path = 'Softwares/Anaconda3/envs/torch/bin/python3.9'
                self.ip = "192.168.50.32"
                tensorboard_arg += ' --bind_all'
                tensorboard_arg += ' --load_fast false'
            self.cuda_num = 4
        elif 'NVIDIA A10' == current_gpu:
            home = "/mnt/18t/pengxin/"
            # __python_path = 'Softwares/Anaconda3/envs/torch/bin/python3.8'
            __python_path = 'Softwares/Anaconda3/envs/torch1110P/torch1110/bin/python3.9'
            self.cuda_num = 8
            self.ip = "192.168.50.33"
            tensorboard_arg += '--host {}'.format(self.ip)
        elif 'NVIDIA GeForce RTX 3090' == current_gpu:
            if os.path.exists('/xlearning/pengxin/Checkpoints/3090.txt'):
                # self.ip = "192.168.50.34"
                # home = "/xlearning/pengxin/"
                # __python_path = 'Software/Anaconda/envs/torch/bin/python3.9'  # Env for FC projects
                # __python_path = 'Software/Anaconda/envs/torch3617/bin/python3.7'  # Env for Temp projects
                # __python_path = 'Software/Anaconda/envs/torch171P37/bin/python3.7'  # Env for FC projects 2.
                # __python_path = 'Software/Anaconda/envs/torch1110/bin/python3.9'  # Env for FC projects in the future.

                self.ip = "192.168.49.44"
                home = "/xlearning/pengxin/"
                __python_path = '/xlearning/pengxin/Software/Anaconda/envs/torch1110/bin/python'
            elif os.path.exists('/xlearning/pengxin/Checkpoints/3090_2.txt'):
                self.ip = "192.168.49.47"
                home = "/xlearning/pengxin/"
                __python_path = '/xlearning/pengxin/Softwares/anaconda3/envs/torch1120/bin/python'
            else:
                self.ip = "pengxin"
                home = "D:\\VirtualMachine"
                __python_path = 'D:/Pengxin/Softwares/anaconda3/envs/torch1121/python.exe'
            self.cuda_num = 1
            tensorboard_arg += ' --load_fast false'
            tensorboard_arg += ' --bind_all'
        else:  # limit@andy-lab
            # noinspection PyUnusedLocal
            home = "~/Pengxin/"
            __python_path = 'python'
            raise NotImplementedError("Unknown Gpu '{}'".format(current_gpu))
        self.tensorboard_arg = tensorboard_arg
        self.home = home
        self.python_path = os.path.join(home, __python_path)

        self.checkpoint_storage_name = checkpoint_storage_name
        self.model_name = model_name
        self.code_run_set_name = code_run_set_name

    def get_dataset_path(self, dataset):
        current_gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        if 'NVIDIA GeForce GTX 1050' == current_gpu:
            return "D:\\L\\Deal\\Dataset\\{}".format(dataset)
        if os.path.exists(os.path.join(self.home, "Data/ImageClassification/", dataset)):
            return os.path.join(self.home, "Data/ImageClassification/", dataset)
        else:
            return os.path.join(self.home, "Data/", dataset)

    def get_code_path(self, code_fold_name=None, level=3):
        """
        /mnt/18t/pengxin/Codes/0814ver/run_set1/Test
        /mnt/18t/pengxin/0    /1      /2       /3

        """
        path_components = ['Codes', self.checkpoint_storage_name, self.code_run_set_name, code_fold_name]
        path = self.home
        for i in range(level + 1):
            assert path_components[i] is not None
            path = os.path.join(path, path_components[i])
        return path

    def get_checkpoint_path(self, code_fold_name=None, level=4, checkpoint_epoch=-1):
        """
        /mnt/18t/pengxin/Checkpoints/SimCLR /0731   /abc/010.checkpoint
        /mnt/18t/pengxin/0          /1      /2      /3  /4
        """
        if code_fold_name is not None:
            code_fold_name = self.clean_path(path=code_fold_name)
        path_components = ['Checkpoints', self.model_name, self.checkpoint_storage_name,
                           code_fold_name, 'epoch{:03d}.checkpoint'.format(checkpoint_epoch)]
        path = self.home
        for i in range(level + 1):
            assert path_components[i] is not None
            path = os.path.join(path, path_components[i])
        return path

    @staticmethod
    def clean_path(path):
        import Utils.ReOperator as ReOperator
        path = ReOperator.ReOperator(string=path).sub(r'_Train\(.*', '')
        path = ReOperator.ReOperator(string=path).sub(r'_Train_.*', '')
        path = ReOperator.ReOperator(string=path).sub(r'_Infer.*', '')
        return path


if __name__ == '__main__':
    pass
