from MainLauncher import path_operator
from Utils import Launcher
from Utils.ConfigOperator import ConfigOperator


def main():
    class C2(ConfigOperator):
        def get_name(self, *args, **kwargs):
            return '_QueueLog'

    Launcher.Launcher(
        path_operator=path_operator,
        env_config='CUBLAS_WORKSPACE_CONFIG=:4096:8 LD_LIBRARY_PATH=/mnt/18t/pengxin/Softwares/Anaconda3/envs/torch1110P/torch1110/lib'
    ).launch(
        run_file='MainLauncher.py',
        cfg=C2(cuda='0'),
        model_name='Train',
        safe_mode=False
    )


if __name__ == '__main__':
    main()
