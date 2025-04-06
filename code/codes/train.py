import argparse
import logging
import math
import os
import random
import sys
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from tqdm import tqdm

# from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)
import torch
from thop import profile

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    # os.environ['RANK'] = str(0)
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str,default="options/dehazing/train/nasde.yml", help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings  是否选择分布式训练
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######
    print('1')
    #### loading resume state if exists 加载参数
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()

        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None
    print('2')
    #### mkdir and loggers 创建目录和日志记录
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            # os.system("rm ./log")
            # os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from tensorboardX import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    print('3')
    #### create train and val dataloader
    dataset_ratio = 1000  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        print(phase,"=",dataset_opt)
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )

            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt)
   #  print(model)
    device = model.device
    # input1 = torch.randn(1, 3, 256, 256)
    # flops, params = profile(model, inputs=(input1,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    # total_epochs = total_epochs * 5

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    print(model)
    sde.set_model(model.model)
    print('sed = ',opt)

    scale = opt['degradation']['scale']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)
    print('5')


    for epoch in range(start_epoch, total_epochs ):
    # for epoch in range(0, 61):
        print('epoch=' ,epoch,'/',total_epochs + 1,'    current_step = ',current_step)
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        # pbar = tqdm(train_loader, total=len(train_loader))
        for _, train_data in enumerate(train_loader):
            # pbar.set_description(f'Epoch [{epoch}/{total_epochs + 1}]')
            current_step += 1

            if current_step > total_iters:
                break

            # print(current_step)
            LQ, GT = train_data["LQ"], train_data["GT"] # 获取数据
            # torchvision.utils.save_image(LQ, "./output/" + str(current_step  + 1) + "LQ.jpg")
            # print(LQ.shape)

            _,_,latent_LQ,prior_LQ= model.encode(LQ.to(device)) # 进行编码，降低图像维度
            # latent_LQ = torch.cat((latent_LQ,prior_LQ),1)

            _,_,latent_GT,prior_GT= model.encode(GT.to(device)) # 同上所示
            # latent_GT = torch.cat((latent_GT, prior_GT), 1)
            # print(latent_LQ.shape)
            timesteps, states = sde.generate_random_states(x0=latent_GT, mu=latent_LQ)
            # print(latent_LQ.shape, timesteps.shape, states.shape)
            # print(timesteps.shape,states.shape) torch.Size([4, 1, 1, 1]) torch.Size([4, 8, 128, 128])

            model.feed_data(states, latent_LQ, latent_GT) # xt, mu, x0
            # print(sde,type(sde))
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)



            # validation, to produce ker_map_list(fake)
            # print(current_step)
            # print(opt["train"]["val_freq"])
            if current_step   % opt["train"]["val_freq"] == 0 and rank <= 0:
                print('测试输出中')
                avg_psnr = 0.0
                idx = 0
                idxs = 0
                for it, val_data in enumerate(val_loader):
                    if it >5 : break
                    LQ, GT = val_data["LQ"], val_data["GT"]
                    LQ.to(device)
                    # print(LQ.shape,GT.shape)
                    with torch.no_grad():
                        x1,x2,latent_LQ,x3= model.encode(LQ.to(device))
                        # latent_LQ = torch.cat((latent_LQ, x3), 1)
                        noisy_state = sde.noise_state(latent_LQ)

                    # valid Predictor
                    model.feed_data(noisy_state, latent_LQ, GT)
                    # print('开始测试')
                    # print('1,', torch.cuda.memory_allocated())
                    model.test(sde=sde,input=LQ.to(device), hidden1=x1,hidden2=x2,hidden3 = x3)
                    # print('2,', torch.cuda.memory_allocated())
                    # print('测试完成')
                    visuals = model.get_current_visuals()
                    # print('3,', torch.cuda.memory_allocated())
                   #  print('数据准备保存')

                    output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                    gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8
                    torchvision.utils.save_image(visuals["Output"], "./output/" + str(idx + 1) + "out.jpg")
                    torchvision.utils.save_image(visuals["GT"], "./output/" + str(idx + 1) + "GT.jpg")



                    # calculate PSNR
                    p = util.calculate_psnr(output, gt_img)
                    avg_psnr += p
                    print('psnr=',p)
                    idx += 1


                    # break
                    torch.cuda.empty_cache()
                for it, val_data in enumerate(train_loader):
                    if it > 3: break
                    LQ, GT = val_data["LQ"], val_data["GT"]
                    LQ.to(device)
                    # print(LQ.shape,GT.shape)
                    with torch.no_grad():
                        x1, x2, latent_LQ, x3 = model.encode(LQ.to(device))
                        # latent_LQ = torch.cat((latent_LQ, x3), 1)
                        noisy_state = sde.noise_state(latent_LQ)

                    # valid Predictor
                    model.feed_data(noisy_state, latent_LQ, GT)
                    # print('开始测试')
                    # print('1,', torch.cuda.memory_allocated())
                    model.test(sde=sde, input=LQ.to(device), hidden1=x1, hidden2=x2, hidden3=x3)
                    # print('2,', torch.cuda.memory_allocated())
                    # print('测试完成')
                    visuals = model.get_current_visuals()
                    # print('3,', torch.cuda.memory_allocated())
                    #  print('数据准备保存')

                    output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                    gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8
                    torchvision.utils.save_image(visuals["Output"], "./output/" + str(idxs + 1) + "train_out.jpg")
                    torchvision.utils.save_image(visuals["GT"], "./output/" + str(idxs + 1) + "train_GT.jpg")

                    # calculate PSNR
                    p = util.calculate_psnr(output, gt_img)
                    avg_psnr += p
                    print('train_psnr=', p)
                    idxs += 1

                avg_psnr = avg_psnr / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

            #     # log
            #     logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
            #     logger_val = logging.getLogger("val")  # validation logger
            #     logger_val.info(
            #         "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
            #             epoch, current_step, avg_psnr
            #         )
            #     )
            #     print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
            #             epoch, current_step, avg_psnr
            #         ))
            #     # tensorboard logger
            #     if opt["use_tb_logger"] and "debug" not in opt["name"]:
            #         tb_logger.add_scalar("psnr", avg_psnr, current_step)
            #
            # if error.value:
            #     sys.exit(0)
            # #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
