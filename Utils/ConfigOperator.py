import os

import yaml
from easydict import EasyDict


# class ConfigOperator:
#     def __init__(self,cuda, yaml_list=None, **kwargs):
#         self.config = EasyDict()
#         self.cuda = cuda
#         if yaml_list is not None:
#             for yaml_path in yaml_list:
#                 with open(yaml_path, 'r') as stream:
#                     config = yaml.safe_load(stream)
#                 self.add_kwargs(**config)
#         self.add_kwargs(**kwargs)
#
#     def add_kwargs(self, **kwargs):
#         for k, v in kwargs.items():
#             self.config[k] = v
#
#     def get_config(self, *args, **kwargs):
#         config = ''
#         for k, val in self.config.items():
#             # k = k.replace('_', '-')
#             if isinstance(val, bool):
#                 if val:
#                     config += '--{} '.format(k)
#             else:
#                 config += '--{} {} '.format(k, val)
#         return config
#
#     def get_name(self, *args, **kwargs):
#         return "{}".format(self.get_config())
#
#     def show_config(self):
#         print(self.config)
#         # print(self.config.setup)
#         # print(self.config.backbone)
#         # print(self.config.model_kwargs)
#         # print(self.config.model_kwargs.head)
class ConfigOperator:
    def __init__(self, cuda, yaml_list=None, **kwargs):
        self.config = EasyDict()
        self.cuda = cuda
        if yaml_list is not None:
            for yaml_path in yaml_list:
                with open(yaml_path, 'r') as stream:
                    config = yaml.safe_load(stream)
                self.add_kwargs(**config)
        self.add_kwargs(**kwargs)
        self.config = EasyDict(dict(sorted(self.config.items())))

    def add_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            if v == '':
                continue
            self.config[k] = v

    def get_config(self, for_path=False, *args, **kwargs):
        config = ''
        for k, val in self.config.items():
            if isinstance(val, bool):
                if val:
                    config += ' --{}'.format(k)
            elif isinstance(val, str) and (len(val.split('\\')) > 1 or len(val.split('/')) > 1):
                if for_path:
                    config += ' --{} {}'.format(
                        k,
                        os.path.join(*val.replace('\\', '/').split('/')[-1:]).replace('/', '@')[:8],
                    )
                else:
                    config += ' --{} \"{}\"'.format(k, val)
            else:
                config += ' --{} {}'.format(k, val)
        return config

    def get_name(self, *args, **kwargs):
        return "{}".format(self.get_config(for_path=True))

    def show_config(self):
        print(self.config)
        # print(self.config.setup)
        # print(self.config.backbone)
        # print(self.config.model_kwargs)
        # print(self.config.model_kwargs.head)


def main():
    config = ConfigOperator(yaml_list=['/home/pengxin/Temp/PythonTemp/SimCLR/configs/Config.yaml'])
    config.show_config()
    # print(config.config.a)

    config.config['b'] = 1
    config.show_config()

    config.config.c = 1
    config.show_config()
    print(config.config.keys())
    print('batch_size' in config.config.keys())


if __name__ == '__main__':
    main()
