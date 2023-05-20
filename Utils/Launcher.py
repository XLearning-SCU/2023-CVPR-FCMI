import os
import subprocess
import time
import warnings

import numpy as np

import Utils.DirectoryOperator as DirectoryOperator
from Utils.Mail import send


class SubprocessOperator:
    def __call__(self, cmd):
        print(cmd)
        if not DirectoryOperator.TestMode:
            subprocess.call(cmd, shell=True)


class SystemCheckOperator(SubprocessOperator):
    def __init__(self):
        super(SystemCheckOperator, self).__init__()

    def check_status(self):
        self.__call__(cmd='nvidia-smi')
        print()
        self.__call__(cmd='ps -ef|grep python')
        print()
        self.__call__(cmd='ps -ef|grep pengxin')
        print()
        self.__call__(cmd='top -n 1')

    def check_process_cwd(self, pid=None):
        if pid is not None:
            self.__call__(cmd='cd /proc/{} &&  ls -l cwd'.format(pid))

    def check_processor(self):
        self.__call__(cmd='cat /proc/cpuinfo |grep "processor"')

    def check_disk(self):
        self.__call__(cmd='lsblk -d -o name,rota')
        print()
        self.__call__(cmd='df -h')
        print()
        self.__call__(cmd='cd {} && du -h --max-depth=1'.format('./'))

    def kill(self, pid=None):
        if pid is not None:
            self.__call__(cmd='kill {}'.format(pid))


class Launcher(SubprocessOperator):
    def __init__(self, path_operator, env_config=''):
        self.path_operator = path_operator
        self.env_config = env_config

    def show_tensorboard(self, path_to_runs):
        python_path = self.path_operator.python_path
        tensorboard_path = os.path.join(os.path.dirname(python_path), 'tensorboard')
        # self.__call__(cmd='find \'{}\' | grep tfevents'.format(path_to_runs))
        # self.__call__(cmd='{} {} --inspect --logdir \'{}\''.format(python_path, tensorboard_path, path_to_runs))
        self.__call__(cmd='{} {} --logdir \'{}\' {}'.format(
            python_path, tensorboard_path, path_to_runs, self.path_operator.tensorboard_arg))

    # def clear_run_set(self):
    #     DirectoryOperator.FoldOperator(directory=self.path_operator.get_code_path(level=2)).clear(
    #         delete_root=False,
    #         not_to_delete_file=self.safe_mode,
    #     )

    def launch(self, run_file, cfg, safe_mode=True, model_name='Train'):
        """

        :param run_file:
        :param cfg:
        :param safe_mode:
        :param model_name:
        :return:

        ./TrainCode
        ./TrainBoard
        ./TrainCheckpoint
        ./Train.txt
        .
        .
        .
        """
        fold_path = self.path_operator.get_code_path(code_fold_name=cfg.get_name(), level=3)
        # if clear_board:
        #     DirectoryOperator.FoldOperator(directory=os.path.join(fold_path, model_name + clear_board)).clear(
        #         delete_root=False,
        #         not_to_delete_file=safe_mode,
        #     )
        # if not safe_mode:
        #     DirectoryOperator.FoldOperator(directory=fold_path).clear(
        #         delete_root=False,
        #         not_to_delete_file=safe_mode,
        #     )
        if os.path.exists(os.path.join(fold_path, 'Checkpoints')):
            warnings.warn('There are some checkpoints in "{}".'.format(fold_path))
        code_root = os.path.join(fold_path, '{}Code'.format(model_name))
        DirectoryOperator.FoldOperator(directory='./').copy(dst_fold=code_root, not_to_delete_file=safe_mode)
        python_cmd = '{} -u {} {}'.format(
            self.path_operator.python_path,
            run_file,
            cfg.get_config(),
        )
        self.__call__(
            cmd="cd '{code_root:}' && {append_config:} nohup {python_cmd:} > '{txt_path:}.txt' 2>&1 &".format(
                code_root=code_root,
                append_config=self.env_config + 'CUDA_VISIBLE_DEVICES={}'.format(cfg.cuda),
                python_cmd=python_cmd,
                txt_path=os.path.join(fold_path, model_name),
            )
        )

    def quick_launch(self, settings, config_operator, run_file='main.py', clear_fold=False, safe_mode=False):
        """
        :param run_file:
        :param settings: [cuda, [yaml_list], arg_dict]
        :param clear_fold:
        :param safe_mode:
        :param config_operator:
        :return:
        """
        for cuda, yaml_list, arg_dict in settings:
            cfg = config_operator(
                yaml_list=yaml_list,
                cuda=cuda,
                **arg_dict,
            )
            if clear_fold:
                DirectoryOperator.FoldOperator(
                    directory=self.path_operator.get_code_path(code_fold_name=cfg.get_name(), level=3)
                ).clear(
                    delete_root=False,
                    not_to_delete_file=safe_mode,
                )
            self.launch(
                run_file=run_file,
                cfg=cfg,
                safe_mode=safe_mode,
            )

    #
    # def train_cfg(self, cfg, run_file, clear_board=True):
    #     self.launch(
    #         code_root=os.path.join(fold_path, '{}Code'.format(model)),
    #         cuda=cfg.cuda,
    #         txt_path=os.path.join(fold_path, model),
    #         config=cfg.get_config(),
    #         run_file=run_file
    #     )

    # def infer_cfg(self, cfg, run_file, fold_path, clear_board):
    #     if clear_board:
    #         DirectoryOperator.FoldOperator(directory=os.path.join(fold_path, 'InferBoard')).clear(
    #             delete_root=False,
    #             not_to_delete_file=self.safe_mode,
    #         )
    #     model = 'Infer'
    #     self.launch(
    #         code_root=os.path.join(fold_path, '{}Code'.format(model)),
    #         cuda=cfg.cuda,
    #         txt_path=os.path.join(fold_path, model),
    #         config=cfg.get_config(),
    #         run_file=run_file
    #     )

    # def run_cfg(self, training, **kwargs):
    #     if training:
    #         self.train_cfg(**kwargs)
    #     else:
    #         self.infer_cfg(**kwargs)


class ProcessQueue:
    def __init__(self, size, work_root, room_size=999):
        self.size = min(size, room_size)
        self.work_root = work_root
        self.subprocess_root = os.path.join(self.work_root, 'Subprocess')
        self.count = 0
        self.start_time = time.time()
        self.interval = 3000
        self.room_size = room_size

    def enqueue(self):
        if os.path.exists(self.subprocess_root):
            dir_list = os.listdir(self.subprocess_root)
            cnt = len(dir_list)
            if self.room_size == self.size and cnt > 0:
                time.sleep(240)
        while True:
            if os.path.exists(self.subprocess_root):
                dir_list = os.listdir(self.subprocess_root)
                cnt = len(dir_list)
                time_now = time.time()
                for f in dir_list:
                    t = time_now - DirectoryOperator.DirectoryOperator(
                        os.path.join(self.subprocess_root, f)
                    ).modification_time()
                    if t > self.interval:
                        send(
                            mail_title="Pengxin's Mail Notification",
                            mail_text="One of your code has been running for {interval:.02f}s, is there something wrong?\n\t--From {sur}".format(
                                interval=t,
                                sur=self.subprocess_root
                            )
                        )
                        self.interval *= 2
                        break
            else:
                cnt = 0
            # print(cnt)
            # print(os.listdir(self.subprocess_root))
            if cnt < self.size:
                time_str = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
                fn = os.path.join(
                    self.subprocess_root,
                    '{}_{:03d}{:03d}.Process'.format(time_str, self.count, cnt)
                )

                print('{time} Enqueue {fn}'.format(time=time_str,
                                                   fn=fn))
                DirectoryOperator.FileOperator(fn).write(data='...')
                self.count += 1
                break
            else:
                time.sleep(10)

    def dequeue(self):
        fn = np.sort(os.listdir(self.subprocess_root))[0]
        print('{time} Dequeue {fn}'.format(time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), fn=fn))
        DirectoryOperator.FileOperator(os.path.join(self.subprocess_root, fn)).remove()

    def close(self):
        self.size = 1
        self.enqueue()
        self.dequeue()
        total_time = time.time() - self.start_time
        pro_count = self.count - 1
        send(
            mail_title="Pengxin's Mail Notification",
            mail_text="Your code finished in {:.02f}s ({}pro * {:.02f}s/pro).\n\t--From {}".format(
                total_time,
                pro_count,
                total_time / pro_count,
                self.subprocess_root
            )
        )


class QueueLauncher(Launcher):
    def __init__(self, queue: ProcessQueue, **kwargs):
        self.queue = queue
        super(QueueLauncher, self).__init__(**kwargs)

    def quick_launch(self, settings, config_operator, run_file='main.py', clear_fold=False, safe_mode=False):
        """
        :param run_file:
        :param settings: [cuda, [yaml_list], arg_dict]
        :param clear_fold:
        :param safe_mode:
        :param config_operator:
        :return:
        """
        for cuda, yaml_list, arg_dict in settings:
            self.queue.enqueue()
            cfg = config_operator(
                yaml_list=yaml_list,
                cuda=cuda,
                **arg_dict,
            )
            if clear_fold:
                DirectoryOperator.FoldOperator(
                    directory=self.path_operator.get_code_path(code_fold_name=cfg.get_name(), level=3)
                ).clear(
                    delete_root=False,
                    not_to_delete_file=safe_mode,
                )
            self.launch(
                run_file=run_file,
                cfg=cfg,
                safe_mode=safe_mode,
            )
        self.queue.close()
