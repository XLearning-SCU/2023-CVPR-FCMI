# import os
import warnings
# from typing import NewType
from DataSetMaster import dataset
# import loss
import networkExplicitDecoupleShuffle
import argparse
import numpy as np
import torch
# torch.cuda.set_device(0)
import random
# os.environ['CUDA_VISIBLE_DEVICE']='0'
from MainLauncher import path_operator, UnDeterministic, Queue

# torch.use_deterministic_algorithms(True)
# torch.autograd.set_detect_anomaly(True)  # auto detect failed to gride
# CUBLAS_WORKSPACE_CONFIG=:4096:8
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='Office')  # MNISTUSPS ReverseMNIST Office HAR MTFL
    parser.add_argument("--MnistTrain", default=1, type=int)
    parser.add_argument("--batch_size", help="batch size", default=512, type=int)
    parser.add_argument("--train_epoch", help="training epochs", default=300, type=int)

    parser.add_argument('--NetVersion', default='Shuffle', type=str)  # New, Old, Explicit, Shuffle
    parser.add_argument('--GroupWiseEncoder', action='store_true')
    parser.add_argument('--GroupWiseDecoder', default=1, type=int)
    parser.add_argument('--WithFeatureB', action='store_true')
    parser.add_argument('--FeatureType',
                        default='GlT_GaussainlizeAndTanh')  # Default Sigmoid Tanh Normalize Gaussainlize GlS_GaussainlizeAndSigmoid GlT_GaussainlizeAndTanh
    parser.add_argument('--BatchNormType', default='1001')  # 0000 -> 1111 1:Norm
    parser.add_argument('--RepresentationType', default='Normalize')  # Normalize None Relu
    parser.add_argument('--ActivationType',
                        default='Tanh')  # None Sigmoid Tanh Normalize Gaussainlize GlS_GaussainlizeAndSigmoid GlT_GaussainlizeAndTanh
    parser.add_argument("--representation_dim", help="", default=0, type=int)

    parser.add_argument('--RunInManuel', action='store_true')
    parser.add_argument("--VisualFreq", help="", default=20, type=int)
    parser.add_argument("--DrawTSNE", action='store_true')
    parser.add_argument("--DrawUmap", action='store_true')

    parser.add_argument("--WarmAll", default=20, type=int)
    parser.add_argument("--WarmBalance", default=0, type=int)
    parser.add_argument("--WarmConsistency", default=0, type=int)
    parser.add_argument("--WarmUpProto", default=0, type=int)
    parser.add_argument("--WarmOneHot", default=0, type=int)

    parser.add_argument("--SoftAssignmentTemperatureBalance", help="", default=0.10, type=float)
    parser.add_argument("--SoftAssignmentTemperatureHot", help="", default=0.10, type=float)
    parser.add_argument("--SoftAssignmentTemperatureSelfCons", help="", default=0.20, type=float)
    parser.add_argument("--Reconstruction", help="", default=1.0, type=float)
    parser.add_argument("--ReconstructionEpoch", default=999, type=int)
    parser.add_argument("--GlobalBalanceLoss", help="", default=0.0, type=float)
    parser.add_argument("--InfoGlobalLoss", help="", default=0.0, type=float)
    parser.add_argument("--InfoBalanceLoss", help="", default=0.04, type=float)
    parser.add_argument("--InfoFairLoss", help="", default=0.20, type=float)
    # parser.add_argument("--IndependentBalanceLoss", help="", default=0.0, type=float)
    parser.add_argument("--GroupWiseBalanceLoss", help="", default=0.0, type=float)  # row
    parser.add_argument("--ClusterWiseBalanceLoss", help="", default=0.0, type=float)  # line
    parser.add_argument("--BalanceLossType", default='KL')  # KL MSE MAE
    parser.add_argument("--BalanceLossNoDetach", action='store_true')
    parser.add_argument("--OneHot", help="", default=0.04, type=float)

    parser.add_argument("--loss_cons", help="", default=0.0, type=float)
    parser.add_argument("--loss_self_cons", help="", default=0.0, type=float)
    parser.add_argument("--SelfConsLossType", default='Assignment')  # Feature Assignment LogAssignment
    parser.add_argument("--Discriminative", help="", default=0.0, type=float)
    parser.add_argument("--DiscriminativeTest", help="", default=0, type=float)
    parser.add_argument("--reconstruct_all", help="", default=0, type=float)
    parser.add_argument("--Decenc", help="", default=0.0, type=float)

    parser.add_argument("--LearnRate", help="", default=1e-3, type=float)
    parser.add_argument("--LearnRateDecayType", default='None')  # Exp, Cosine
    parser.add_argument("--WeightDecay", default=0, type=float)
    parser.add_argument("--LearnRateWarm", default=0, type=int)
    parser.add_argument("--betas_a", help="", default=0.9, type=float)
    parser.add_argument("--betas_v", help="", default=0.999, type=float)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--resume", default='')

    parser.add_argument("--LambdaClu", help="alpha in our main paper Eq. 12", default=0.04, type=float)
    parser.add_argument("--LambdaFair", help="beta in our main paper Eq. 12", default=0.20, type=float)

    args = parser.parse_args()
    if args.RunInManuel:
        warnings.warn('RunInManuel')
    if args.dataset == 'MTFL':
        args.representation_dim = 5
    if args.LambdaClu:
        args.InfoBalanceLoss = args.LambdaClu
        args.OneHot = args.LambdaClu
    if args.LambdaFair:
        args.InfoFairLoss = args.InfoFairLoss

    print('=======Arguments=======')
    print(path_operator.python_path)
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    if not UnDeterministic:
        Torch171 = False
        if Torch171:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.set_deterministic(True)
            torch.backends.cudnn.deterministic = True
        else:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
    else:
        warnings.warn('Not deterministic')
    if args.train_epoch >= 0:
        train_loader, test_loader, class_num = dataset.get_dataloader(
            dataset=args.dataset, batch_size=args.batch_size, path=path_operator.get_dataset_path, args=args)
        if args.NetVersion == 'New':
            raise NotImplementedError('args.NetVersion')
            # net_edition = network_new
        elif args.NetVersion == 'Old':
            raise NotImplementedError('args.NetVersion')
            # net_edition = network
        elif args.NetVersion == 'Explicit':
            raise NotImplementedError('args.NetVersion')
            # net_edition = networkExplicitDecouple
        elif args.NetVersion == 'Shuffle':
            net_edition = networkExplicitDecoupleShuffle
        else:
            raise NotImplementedError('args.NetVersion')

        if args.dataset == 'Office':
            net = net_edition.NetFCN(class_num=class_num, input_dim=2048, group_num=2, args=args).cuda()
        elif args.dataset == 'HAR':
            net = net_edition.NetFCN(class_num=class_num, input_dim=561, group_num=30, args=args).cuda()
        elif args.dataset == 'MTFL':
            net = net_edition.NetFCN(class_num=class_num, input_dim=2048, group_num=2, args=args).cuda()
        else:
            net = net_edition.Net(class_num=class_num, group_num=2, args=args).cuda()

        # from torchsummary import summary
        # summary(net, (1,28,28), batch_size=512)
        net.run(epochs=args.train_epoch,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                args=args)
    print('Ending...')
    Queue.dequeue()
    # file_path = './net_multiAE_' + str(
    #     args.dataset) + '_parameter_seed_' + str(args.seed) + '.pth'
    # torch.save(net.state_dict(), file_path)
    # print('Save the trained model to', file_path)

#  conda activate torch && cd /xlearning/pengxin/Temp/Python/FC3 && CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=5 python main.py
