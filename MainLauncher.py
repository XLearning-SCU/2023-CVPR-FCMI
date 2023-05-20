import os
import time
import warnings

import numpy as np

import Utils.ConfigOperator as ConfigOperator
import Utils.PathPresettingOperator as PathPresettingOperator
import Utils.Launcher as Launcher
import Utils.DirectoryOperator as DirectoryOperator

DirectoryOperator.TestMode = False
CodeTest = False
path_operator = PathPresettingOperator.PathOperator(
    model_name='FairClustering',
    checkpoint_storage_name='230520',
    code_run_set_name='RunSet0520_RePo5',
)
UnDeterministic = False
LimitKmeans = True  # Lik
DrawMax = 10000


# class SConfigOperator(ConfigOperator.ConfigOperator):
#     def get_name(self):
#         data_name = self.config['dataset']
#         if 'NetVersion' in self.config.keys() and self.config['NetVersion'] == 'Shuffle' \
#                 and 'MnistTrain' in self.config.keys() and self.config['MnistTrain'] \
#                 and (self.config['dataset'] == 'MNISTUSPS' or self.config['dataset'] == 'ReverseMNIST'):
#             data_name += 'train'
#         frame_setting = ''
#         frame_setting += 'Net{}'.format(self.config['NetVersion'])
#         if LimitKmeans:
#             frame_setting += 'Lik'
#         if CodeTest:
#             frame_setting += 'CodeTest'
#         if 'GroupWiseEncoder' in self.config.keys() and self.config['GroupWiseEncoder']:
#             frame_setting += 'GEnc'
#         if 'GroupWiseDecoder' in self.config.keys() and self.config['GroupWiseDecoder']:
#             frame_setting += 'GDec'
#         if 'WithFeatureB' in self.config.keys() and self.config['WithFeatureB']:
#             frame_setting += 'FeaB'
#
#         if self.config['dataset'] == 'Office' or self.config['dataset'] == 'MTFL':
#             if 'FeatureType' in self.config.keys() and self.config['FeatureType'] != 'Default':
#                 frame_setting += 'Fea{}'.format(self.config['FeatureType'][:3])
#         if self.config['dataset'] == 'Office' or self.config['dataset'] == 'MTFL' or self.config['dataset'] == 'HAR':
#             if 'BatchNormType' in self.config.keys() and self.config['BatchNormType'] != '1001':
#                 frame_setting += 'Bat{}'.format(self.config['BatchNormType'])
#         if 'RepresentationType' in self.config.keys() and self.config['RepresentationType'] != 'Normalize':
#             frame_setting += 'Rep{}'.format(self.config['RepresentationType'])
#         if self.config['dataset'] == 'Office' or self.config['dataset'] == 'MTFL':
#             frame_setting += 'NormRes'
#         if self.config['dataset'] == 'Office' or self.config['dataset'] == 'HAR' or self.config['dataset'] == 'MTFL':
#             if self.config['dataset'] == 'HAR':
#                 self.config['ActivationType'] = 'Tanh'
#             if 'ActivationType' in self.config.keys() and self.config['ActivationType'] != 'None':
#                 frame_setting += 'Act{}'.format(self.config['ActivationType'][:3])
#         if 'representation_dim' in self.config.keys() and self.config['representation_dim'] != 0:
#             frame_setting += 'Rd{:02d}'.format(self.config['representation_dim'])
#         if 'DrawTSNE' in self.config.keys() and self.config['DrawTSNE']:
#             frame_setting += 'DrawTSNE'
#
#         if 'WarmAll' in self.config.keys() and self.config['WarmAll']:
#             frame_setting += 'WarmAll{:02d}'.format(self.config['WarmAll'])
#         if 'WarmUpProto' in self.config.keys() and self.config['WarmUpProto']:
#             frame_setting += 'WarmProto{:02d}'.format(self.config['WarmUpProto'])
#         if 'WarmConsistency' in self.config.keys() and self.config['WarmConsistency']:
#             frame_setting += 'WarmConsistency{:02d}'.format(self.config['WarmConsistency'])
#         if 'WarmBalance' in self.config.keys() and self.config['WarmBalance']:
#             frame_setting += 'WarmBalance{:02d}'.format(self.config['WarmBalance'])
#         if 'WarmOneHot' in self.config.keys() and self.config['WarmOneHot']:
#             frame_setting += 'WarmOneHot{:02d}'.format(self.config['WarmOneHot'])
#
#         if 'Reconstruction' in self.config.keys() and self.config['Reconstruction'] != 1.0:
#             frame_setting += 'Rec{:04.02f}'.format(self.config['Reconstruction'])
#         if 'ReconstructionEpoch' in self.config.keys() and self.config['ReconstructionEpoch'] != 999:
#             frame_setting += 'RecE{:03d}'.format(self.config['ReconstructionEpoch'])
#         if ('GlobalBalanceLoss' in self.config.keys() and self.config['GlobalBalanceLoss']) or (
#                 'GroupWiseBalanceLoss' in self.config.keys() and self.config['GroupWiseBalanceLoss']) or (
#                 'ClusterWiseBalanceLoss' in self.config.keys() and self.config['ClusterWiseBalanceLoss']):
#             if 'BalanceLossType' in self.config.keys() and self.config['BalanceLossType'] != 'KL':
#                 frame_setting += '{}'.format(self.config['BalanceLossType'])
#         if 'GlobalBalanceLoss' in self.config.keys() and self.config['GlobalBalanceLoss']:
#             frame_setting += 'GlobalBalance{:04.02f}'.format(self.config['GlobalBalanceLoss'])
#         if 'InfoGlobalLoss' in self.config.keys() and self.config['InfoGlobalLoss']:
#             frame_setting += 'InfoGlobalLoss{:04.02f}'.format(self.config['InfoGlobalLoss'])
#         if 'InfoBalanceLoss' in self.config.keys() and self.config['InfoBalanceLoss']:
#             frame_setting += 'InfoBalanceLoss{:04.02f}'.format(self.config['InfoBalanceLoss'])
#         if 'InfoFairLoss' in self.config.keys() and self.config['InfoFairLoss']:
#             frame_setting += 'InfoFairLoss{:04.02f}'.format(self.config['InfoFairLoss'])
#
#         if 'GroupWiseBalanceLoss' in self.config.keys() and self.config['GroupWiseBalanceLoss']:
#             frame_setting += 'GroupBalance{:04.02f}'.format(self.config['GroupWiseBalanceLoss'])
#         if 'ClusterWiseBalanceLoss' in self.config.keys() and self.config['ClusterWiseBalanceLoss']:
#             frame_setting += 'ClusterBalance{:04.02f}'.format(self.config['ClusterWiseBalanceLoss'])
#         if 'SoftAssignmentTemperatureBalance' in self.config.keys() and \
#                 self.config['SoftAssignmentTemperatureBalance'] != 1.0:
#             frame_setting += 'Tb{:04.02f}'.format(self.config['SoftAssignmentTemperatureBalance'])
#
#         if 'OneHot' in self.config.keys() and self.config['OneHot']:
#             frame_setting += 'OneHot{:04.02f}'.format(self.config['OneHot'])
#             if 'SoftAssignmentTemperatureHot' in self.config.keys() and \
#                     self.config['SoftAssignmentTemperatureHot'] != 1.0:
#                 frame_setting += 'Th{:04.02f}'.format(self.config['SoftAssignmentTemperatureHot'])
#
#         if 'loss_cons' in self.config.keys() and self.config['loss_cons']:
#             frame_setting += 'Proto{:04.02f}'.format(self.config['loss_cons'])
#         if 'loss_self_cons' in self.config.keys() and self.config['loss_self_cons']:
#
#             if 'SelfConsLossType' in self.config.keys() and self.config['SelfConsLossType'] != 'Feature':
#                 if self.config['SelfConsLossType'] == 'LogAssignment':
#                     frame_setting += 'LogAss'
#                 else:
#                     frame_setting += '{}'.format(self.config['SelfConsLossType'][:3])
#             frame_setting += 'Consistency{:04.02f}'.format(self.config['loss_self_cons'])
#             if 'SoftAssignmentTemperatureSelfCons' in self.config.keys() and \
#                     self.config['SoftAssignmentTemperatureSelfCons'] != 1.0:
#                 frame_setting += 'Tc{:04.02f}'.format(self.config['SoftAssignmentTemperatureSelfCons'])
#         if 'Discriminative' in self.config.keys() and self.config['Discriminative']:
#             frame_setting += 'Discri{:04.02f}'.format(self.config['Discriminative'])
#         if 'reconstruct_all' in self.config.keys() and self.config['reconstruct_all']:
#             frame_setting += 'RecAll{:04.02f}'.format(self.config['reconstruct_all'])
#         if 'Decenc' in self.config.keys() and self.config['Decenc'] != 0:
#             frame_setting += 'Decenc{:04.02f}'.format(self.config['Decenc'])
#         if 'LearnRate' in self.config.keys() and self.config['LearnRate'] != 1e-3:
#             frame_setting += 'LR{}'.format(str(self.config['LearnRate']))
#         if 'LearnRateDecayType' in self.config.keys() and self.config['LearnRateDecayType'] != 'None':
#             frame_setting += '{}Decay'.format(self.config['LearnRateDecayType'][:3])
#         if 'WeightDecay' in self.config.keys() and self.config['WeightDecay'] != 0:
#             frame_setting += 'WeightDecay{}'.format(self.config['WeightDecay'])
#         if 'LearnRateWarm' in self.config.keys() and self.config['LearnRateWarm'] != 0:
#             frame_setting += 'LRwarm{:02d}'.format(self.config['LearnRateWarm'])
#         if 'betas_a' in self.config.keys() and self.config['betas_a'] != 0.9:
#             frame_setting += 'BetasA{:03.01f}'.format(self.config['betas_a'])
#         if 'betas_v' in self.config.keys() and self.config['betas_v'] != 0.999:
#             frame_setting += 'BetasV{:05.03f}'.format(self.config['betas_v'])
#         if 'seed' in self.config.keys() and self.config['seed'] != 0:
#             frame_setting += 'Seed{:04d}'.format(self.config['seed'])
#         gpu_setting = 'G{}'.format(self.cuda)
#         gpu_setting += 'B{:04d}'.format(self.config['batch_size'])
#         gpu_setting += path_operator.python_path.split('/')[-3]
#
#         fold_name = '_'.join([data_name, frame_setting, gpu_setting])
#         return fold_name


cudas = [0, 1, 2, 3]

Queue = Launcher.ProcessQueue(
    size=2,
    room_size=len(cudas),
    work_root=os.path.join(path_operator.get_code_path(level=2), '_QueueLog')
)


# recommend InfoBalanceLoss in 0.01,0.02,0.03,0.04  InfoFairLoss in  0.10 0.15 0.20
def get_settings():
    # 'batch_size': 512, 'LearnRate': 1e-3 * a / 512
    # 'WarmAll': 9, 'VisualFreq': 10,
    # ['MTFL', 'HAR', 'MNISTUSPS', 'ReverseMNIST', 'Office'],
    # 'MnistTrain': True, 'train_epoch': 200,
    # 'LearnRate': lr, 'WeightDecay': wd
    # 'resume':, 'DrawTSNE': True, 'DrawUmap': True
    # [0, 9116, 2022, 996, 1970, 1971, 123, 1998, 1999, 916]
    a = []

    # a += [
    #     [['QuickConfig/BaseConfig2.yaml'], {
    #         'dataset'                          : ds,
    #         'SoftAssignmentTemperatureBalance' : 0.10, 'InfoBalanceLoss': 0.04, 'InfoFairLoss': 0.20,
    #         'SoftAssignmentTemperatureHot'     : 0.10, 'OneHot': 0.04,
    #         'SoftAssignmentTemperatureSelfCons': 0.2, 'loss_self_cons': 0.00,
    #         'seed'                             : sd, 'VisualFreq': 20, 'train_epoch': 300, 'MnistTrain': True,
    #         'FeatureType'                      : 'GlT_GaussainlizeAndTanh', 'ActivationType': 'Tanh',
    #         'representation_dim'               : 0,
    #     }]
    #     for ds, sd in [
    #         ['MNISTUSPS', 9116],
    #         ['MTFL', 9116],
    #         ['Office', 0],
    #         ['HAR', 9116],
    #         ['ReverseMNIST', 0],
    #     ]
    # ]
    a += [
        [[], {
            'dataset': ds, 'seed': sd, 'resume': rs
        }]
        for ds, sd, rs in [
            # ['MTFL', 9116,'/mnt/18t/pengxin/Checkpoints/FairClustering/SOTA0421/MTFL_NetShuffleLikGDecFeaGlTNormResActTanRd05WarmAll20InfoBalanceLoss0.04InfoFairLoss0.20Tb0.10OneHot0.04Th0.10Seed9116_G2B0512torch1110/Checkpoints/Epoch299.checkpoint'],
            # ['Office', 0,'/mnt/18t/pengxin/Checkpoints/FairClustering/SOTA0421/Office_NetShuffleLikGDecFeaGlTNormResActTanWarmAll20InfoBalanceLoss0.04InfoFairLoss0.20Tb0.10OneHot0.04Th0.10_G2B0512torch1110/Checkpoints/Epoch299.checkpoint'],
            # ['HAR', 9116,'/mnt/18t/pengxin/Checkpoints/FairClustering/SOTA0421/HAR_NetShuffleLikGDecActTanWarmAll20InfoBalanceLoss0.04InfoFairLoss0.20Tb0.10OneHot0.04Th0.10Seed9116_G0B0512torch1110/Checkpoints/Epoch299.checkpoint'],
            # ['ReverseMNIST', 0,'/mnt/18t/pengxin/Checkpoints/FairClustering/SOTA0421/ReverseMNISTtrain_NetShuffleLikGDecWarmAll20InfoBalanceLoss0.04InfoFairLoss0.20Tb0.10OneHot0.04Th0.10_G1B0512torch1110/Checkpoints/Epoch299.checkpoint'],
            # ['MNISTUSPS', 9116,'/mnt/18t/pengxin/Checkpoints/FairClustering/SOTA0421/MNISTUSPStrain_NetShuffleLikGDecWarmAll20InfoBalanceLoss0.04InfoFairLoss0.20Tb0.10OneHot0.04Th0.10Seed9116_G2B0512torch1110/Checkpoints/Epoch299.checkpoint'], # 94.5 -> 92.3

            # ['MTFL', 9116, '/xlearning/pengxin/FC_Cps/MTFL/Epoch299.checkpoint'],
            # ['Office', 0, '/xlearning/pengxin/FC_Cps/OFFICE/Epoch299.checkpoint'],
            # ['HAR', 9116, '/xlearning/pengxin/FC_Cps/HAR/Epoch299.checkpoint'],
            ['ReverseMNIST', 0, '/xlearning/pengxin/FC_Cps/REVMNIST/Epoch299.checkpoint'],
            # ['MNISTUSPS', 9116, '/xlearning/pengxin/FC_Cps/MNISTUSPS/Epoch299.checkpoint'],  # 94.5 -> 92.3
        ]
    ]

    cudalen = len(cudas)
    a = [['{}'.format(cudas[i % cudalen])] + it for i, it in enumerate(a)]
    return a


def run(use_queue=True):
    blocking = False
    gun = False
    if UnDeterministic:
        warnings.warn("Not determinstic")
    if blocking:
        warnings.warn("Blocking == 1 only for debug(force synchronous computation)")
    if gun:
        warnings.warn("MKL_THREADING_LAYER=GNU")
    # env_config = '{}LD_LIBRARY_PATH=/xlearning/pengxin/Software/Anaconda/envs/torch1110/lib {}{}'.format(
    env_config = '{}LD_LIBRARY_PATH={} {}{}'.format(
        'CUBLAS_WORKSPACE_CONFIG=:4096:8 ' if not UnDeterministic else '',
        path_operator.python_path.replace('bin/python3.9', 'lib'),
        'CUDA_LAUNCH_BLOCKING=1 ' if blocking else '',
        'MKL_THREADING_LAYER=GNU ' if gun else '',
    )
    if use_queue:
        launcher = Launcher.QueueLauncher(path_operator=path_operator, env_config=env_config, queue=Queue)
    else:
        launcher = Launcher.Launcher(path_operator=path_operator, env_config=env_config)
    launcher.quick_launch(settings=get_settings(), config_operator=ConfigOperator.ConfigOperator, clear_fold=True, safe_mode=False)


def main():
    # print(get_settings2())
    # print(path_operator.get_dataset_path(''))
    run()
    # Launcher.SystemCheckOperator().check_status()
    # Launcher.SystemCheckOperator().kill(3350717)
    # Launcher.SystemCheckOperator().kill(3350715)
    # Launcher.SystemCheckOperator().kill(3109500)
    # Launcher.SystemCheckOperator().kill(1064109)
    # Launcher.SystemCheckOperator().kill(3772438)
    # Launcher.SystemCheckOperator().kill(3772441)
    # Launcher.SystemCheckOperator().kill(3772444)
    # Launcher.SystemCheckOperator().kill(3772447)
    # Launcher.SystemCheckOperator().kill(3772310)
    # Launcher.SystemCheckOperator().kill(3772311)
    # Launcher.SystemCheckOperator().check_process_cwd(3624471)
    # Launcher.SystemCheckOperator().check_process_cwd(3696044)
    # Launcher.Launcher(path_operator=path_operator).tensorboard(
    #     path_to_runs='/xlearning/pengxin/Checkpoints/MoCo/0112/Ex2DxT050S6IdentInc_ImageNet100_Batch0256M3,4Worker14')


if __name__ == '__main__':
    main()
    print('Ending')
# 32
# 3090
# A10
# 31

# 192.168.49.
# 49  titan2
# 44  3090
# 46  titan1
# 48  a10

#########################################
# 使用优化
############
# train 完 自动 infer
# 独立infer
# infer 完 自动邮件
# 注册待运行, 空闲自动挂载

# cd /xlearning/pengxin/Temp/Python/FC3 && CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 LD_LIBRARY_PATH=/xlearning/pengxin/Software/Anaconda/envs/torch171P37/lib compute-sanitizer /xlearning/pengxin/Software/Anaconda/envs/torch171P37/bin/python3.7 -u main.py --batch_size 512 --train_epoch 300 --NetVersion Shuffle --div_loss 0.3 --BalanceTemperature 0.5 --BalanceClusterWise 0.2 --loss_cons 0.5 --loss_self_cons 0.7 --Discriminative 0.5 --reconstruct_all 0.0 --dataset Office
