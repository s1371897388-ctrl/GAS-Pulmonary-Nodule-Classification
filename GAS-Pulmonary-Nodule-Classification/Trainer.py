import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import math
from collections import OrderedDict

import utils.util as utils
from data import create_dataLoader
from models import create_model
from utils.visualizer import Visualizer
from models.self_distillation import SelfDistillationModel, DIYSelfDistillationModel
from models.fusion_module import FusionModule
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
class Trainer():

    def __init__(self, opt, logger):

        self.opt = opt
        self.opt.isTrain = True
        self.logger = logger
        self.visualizer = Visualizer(opt)
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if torch.cuda.is_available() else 'cpu'

        self.epochs = opt.n_epochs
        self.start_epochs = opt.start_epoch
        self.train_batch_size = self.opt.train_batch_size
        self.temperature = self.opt.temperature

        dataLoader = create_dataLoader(opt)
        self.trainLoader = dataLoader.trainLoader
        self.testLoader = dataLoader.testLoader

        class_weights = torch.tensor([0.45, 0.55])
        self.criterion_CE = nn.CrossEntropyLoss(weight=class_weights).to(self.device)

        self.criterion_KL = nn.KLDivLoss(reduction='batchmean').to(self.device)

        self.model_num = opt.model_num
        self.models = []
        self.optimizers = []
        self.schedulers = []
        for i in range(self.model_num):
            model = create_model(opt).to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                  weight_decay=opt.weight_decay,
                                  nesterov=True)
            scheduler = utils.get_scheduler(optimizer, opt)
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

        self.init_self_ditsllation_models()#初始化自蒸馏模型。
        self.init_fusion_module()#初始化融合模块。

        self.leader_model = create_model(self.opt, leader=True, trans_fusion_info=(self.fusion_channel, self.model_num)).to(self.device)#创建领导模型，用于融合多个模型的特征。
        self.leader_optimizer = optim.SGD(self.leader_model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                          weight_decay=self.opt.weight_decay,
                                          nesterov=True)#创建领导模型的优化器。
        self.leader_scheduler = utils.get_scheduler(self.leader_optimizer, opt)#创建领导模型的学习率调度器。



    def init_self_ditsllation_models(self):#初始化自蒸馏模型
        input_size = None  # 初始化 input_size

        if str.startswith(self.opt.dataset, 'LIDC_IDRI'):#根据数据集类型确定输入尺寸，对于cifar数据集，输入尺寸为(1, 3, 32, 32)，对于其他数据集，输入尺寸为(1, 3, 224, 224)。
            input_size = (1, 3, 32, 32)
        # print(f"Dataset type: {self.opt.dataset}, Input size: {input_size}")
        # else:
        #     input_size = (1, 3, 224, 224)
        if input_size is None:
            raise ValueError("input_size is not set. Check dataset type.")

        noise_input = torch.randn(input_size).to(self.device)#生成符合输入尺寸的随机噪声输入noise_input。
        # print("noise_input", noise_input.shape)
        self.models[0](noise_input)
        trans_input = list(self.models[0].total_feature_maps.values())[-1]#将噪声输入传入第一个模型，并获取最后一个特征图trans_input。
        #根据最后一个特征图获取融合通道数和空间尺寸。
        self.fusion_channel = trans_input.size(1)
        self.fusion_spatil = trans_input.size(2)

        self.sd_models = []
        self.sd_optimizers = []
        self.sd_schedulers = []

        for i in range(1, self.model_num):#循环从1到self.model_num-1，创建对应数量的自蒸馏模型。
            #普通学生的模型student1，student2
            #根据不同模型的特性和要求选择不同的自蒸馏模型配置，以最大程度地提高模型性能和泛化能力。
            if str.startswith(self.opt.model, 'DWCGhost'):
            #     sd_model = DIYSelfDistillationModel([456, 312, 168], 2).to(self.device)
            # elif str.startswith(self.opt.model, 'googlenet'):
            #     sd_model = DIYSelfDistillationModel([1024, 832, 480], 2).to(self.device)
            # else:
                sd_model = SelfDistillationModel(input_channel=trans_input.size(1),
                                                 layer_num=len(self.models[0].extract_layers) - 1).to(self.device)
            elif str.startswith(self.opt.model, 'googlenet'):
                sd_model = DIYSelfDistillationModel([1024, 832, 480], 2).to(self.device)
            else:
                sd_model = SelfDistillationModel(input_channel=trans_input.size(1),
                                                 layer_num=len(self.models[0].extract_layers) - 1).to(self.device)

           #————————————————————————————————————————————————————————————————————————————————————————————————
            #为自蒸馏模型创建优化器、学习率调度器

            sd_optimizer = optim.Adam(sd_model.parameters(), weight_decay=self.opt.weight_decay)
            sd_scheduler = utils.get_scheduler(sd_optimizer, self.opt)
            #将创建的自蒸馏模型添加到各自的列表中。
            self.sd_models.append(sd_model)
            self.sd_optimizers.append(sd_optimizer)
            self.sd_schedulers.append(sd_scheduler)
            #leader student模型的通道配置
        if str.startswith(self.opt.model, 'DWCGhost'):
        #     self.sd_leader_model = DIYSelfDistillationModel([456, 312, 168], 2).to(self.device)
        # elif str.startswith(self.opt.model, 'googlenet'):
        #     self.sd_leader_model = DIYSelfDistillationModel([1024, 832, 480], 2).to(self.device)
        # else:
            self.sd_leader_model = SelfDistillationModel(input_channel=trans_input.size(1),
                                                         layer_num=len(self.models[0].extract_layers) - 1).to(self.device)
        elif str.startswith(self.opt.model, 'googlenet'):
            self.sd_leader_model = DIYSelfDistillationModel([1024, 832, 480], 2).to(self.device)
        else:
            self.sd_leader_model = SelfDistillationModel(input_channel=trans_input.size(1),
                                                         layer_num=len(self.models[0].extract_layers) - 1).to(self.device)
        self.sd_leader_optimizer = optim.Adam(self.sd_leader_model.parameters(), weight_decay=self.opt.weight_decay)
        self.sd_leader_scheduler = utils.get_scheduler(self.sd_leader_optimizer, self.opt)

    def init_fusion_module(self):

        self.num_classes = 2#模型的分类数100/////2
#创建了一个融合模块self.fusion_module，使用了名为FusionModule的类。参数包括融合的通道数self.fusion_channel、类别数self.num_classes、空间维度self.fusion_spatil和模型数量model_num。
        self.fusion_module = FusionModule(self.fusion_channel, self.num_classes,
                                          self.fusion_spatil, model_num=self.model_num).to(self.device)
#使用随机梯度下降（SGD）优化器为融合模块的参数创建了优化器self.fusion_optimizer。设置了学习率为self.opt.lr、动量为self.opt.momentum、权重衰减为1e-5、Nesterov动量为True。
        self.fusion_optimizer = optim.SGD(self.fusion_module.parameters(), lr=self.opt.lr, momentum=self.opt.momentum,
                                          weight_decay=1e-5,
                                          nesterov=True)
#融合模块的优化器创建了学习率调度器
        self.fusion_scheduler = utils.get_scheduler(self.fusion_optimizer, self.opt)

    def train(self):

        topk = (1,)#计算准确率时的top-k值，这里表示计算top-1准确率。

        best_acc = [0.0] * self.model_num#best_acc记录每个模型的最佳准确率
        best_epoch = [1] * self.model_num#best_epoch记录对应的epoch。
        best_avg_acc = 0.0#记录平均准确率
        best_ens_acc = 0.0#记录集成准确率
        best_avg_epoch = 1
        best_ens_epoch = 1
        best_fusion_acc = 0.0#记录融合模型的最佳准确率
        best_fusion_epoch = 1
        best_leader_acc = 0.0#领导模型的最佳准确率。
        best_leader_epoch = 1

        for epoch in range(self.start_epochs, self.epochs):#从self.start_epochs开始训练，直到self.epochs为止。

            self.visualizer.reset()#重置可视化工具，可能是为了记录每个epoch的训练状态。

            self.lambda_warmup(epoch)#用于训练初期的学习率预热。
            self.train_with_test(epoch, topk=topk)#进行训练并输出训练过程中的信息。

            _, test_acc, _, test_avg_acc, test_ens_acc = self.test(epoch, topk=topk)#调用test方法，进行测试并返回测试结果，包括每个模型的准确率、平均准确率和集成准确率。

            for i in range(self.model_num):#遍历每个模型，保存当前模型的训练结果，并根据测试准确率更新最佳模型的信息。
                self.save_models(self.models[i], epoch, str(i), self.opt, isbest=False)
                if test_acc[i].avg > best_acc[i]:
                    best_acc[i] = test_acc[i].avg
                    best_epoch[i] = epoch
                    self.save_models(self.models[i], epoch, str(i), self.opt, isbest=True)

            if test_acc[-2].avg > best_fusion_acc:#检查融合模型的测试准确率是否超过历史最佳准确率，如果是，则保存当前融合模型为最佳模型。
                self.save_models(self.fusion_module, epoch, 'fusion', self.opt, isbest=True)
                best_fusion_acc = test_acc[-2].avg
                best_fusion_epoch = epoch

            if test_acc[-1].avg > best_leader_acc:#检查leader——student模型的测试准确率是否超过历史最佳准确率，如果是，则保存当前融合模型为最佳模型。
                self.save_models(self.leader_model, epoch, 'leader', self.opt, isbest=True)
                best_leader_acc = test_acc[-1].avg
                best_leader_epoch = epoch
#检查平均准确率和集成准确率是否超过历史最佳准确率，如果是，则更新最佳准确率和对应的epoch。
            if test_avg_acc.avg > best_avg_acc:
                best_avg_acc = test_avg_acc.avg
                best_avg_epoch = epoch
            if test_ens_acc.avg > best_ens_acc:
                best_ens_acc = test_ens_acc.avg
                best_ens_epoch = epoch




            for i, scheduler in enumerate(self.schedulers):#更新所有模型的学习率调度器，可能是为了在训练过程中动态调整学习率。
                if i > 0:
                    self.sd_schedulers[i-1].step()
                scheduler.step()
            self.sd_leader_scheduler.step()
            self.fusion_scheduler.step()
            self.leader_scheduler.step()
        #输出最佳模型的信息和训练过程中的一些日志信息，包括平均准确率、集成准确率、融合模型和领导模型的最佳准确率和对应的epoch。
        best_msg = 'Best Models: '
        self.logger.info(
            'Best Average/Ensemble Epoch{}:{:.2f}/Epoch{}:{:.2f}'.format(best_avg_epoch, float(best_avg_acc),
                                                                         best_ens_epoch, float(best_ens_acc)))
        for i in range(self.model_num):
            best_msg += 'Epoch {}:{:.2f}/'.format(best_epoch[i], float(best_acc[i]))
        self.logger.info(
            'Model[Fusion]/[Leader] Epoch{}:{:.2f}/Epoch{}:{:.2f}'.format(best_fusion_epoch, float(best_fusion_acc),
                                                                          best_leader_epoch, float(best_leader_acc)))
        self.logger.info(best_msg)

    def train_with_test(self, epoch, topk=(1,)):

        accuracy = []
        losses = []
        ce_losses = []#模型训练过程中的交叉熵损失值
        dml_losses = []#存储模型训练过程中的深度度量损失值。
        diversity_losses = []#存储模型训练过程中的多样性损失值。
        self_distillation_feature_losses = []#存储模型训练过程中的自蒸馏特征损失值。
        self_distillation_attention_losses = []#存储模型训练过程中的自蒸馏注意力损失值。
        self_distillation_losses = []#存储模型训练过程中的自蒸馏总损失值。

        fusion_accuracy = utils.AverageMeter()#创建一个用于计算平均值的对象，用于记录融合模型的准确率。
        fusion_ce_loss = utils.AverageMeter()#创建一个用于计算平均值的对象，用于记录融合模型的交叉熵损失值。
        fusion_ensemble_loss = utils.AverageMeter()#创建一个用于计算平均值的对象，用于记录融合模型的集成损失值。
        fusion_loss = utils.AverageMeter()#记录融合模型的总损失值。

        leader_accuracy = utils.AverageMeter()#记录领导模型的准确率。
        leader_ce_loss = utils.AverageMeter()#记录领导模型的交叉熵损失值。
        leader_ensemble_loss = utils.AverageMeter()
        leader_self_distillation_feature_loss = utils.AverageMeter()
        leader_self_distillation_attention_loss = utils.AverageMeter()
        leader_self_distillation_loss = utils.AverageMeter()
        leader_fusion_loss = utils.AverageMeter()
        leader_trans_fusion_loss = utils.AverageMeter()#用于记录领导模型的转换融合损失值。
        leader_loss = utils.AverageMeter()#于记录领导模型的总损失值。

        average_accuracy = utils.AverageMeter()#记录平均准确率。
        ensemble_accuracy = utils.AverageMeter()#记录集成准确率。

        self.fusion_module.train()#将融合模型设置为训练模式
        self.leader_model.train()#将领导模型设置为训练模式
        for i in range(self.model_num):#循环遍历普通模型列表。
            self.models[i].train()#将当前普通模型设置为训练模式。=
            losses.append(utils.AverageMeter())#用于记录损失值
            ce_losses.append(utils.AverageMeter())#记录交叉熵损失值。
            dml_losses.append(utils.AverageMeter())#用于记录深度度量损失值
            diversity_losses.append(utils.AverageMeter())#用于记录多样性损失值
            self_distillation_feature_losses.append(utils.AverageMeter())#记录自蒸馏特征损失值
            self_distillation_attention_losses.append(utils.AverageMeter())#记录自蒸馏注意力损失值。
            self_distillation_losses.append(utils.AverageMeter())#用于记录自蒸馏总损失值
            accuracy.append(utils.AverageMeter())#记录准确率。

        print_freq = len(self.trainLoader.dataset) // self.opt.train_batch_size // 10#设置打印频率，即每训练完成10%数据就打印一次
        # print_freq = 1  # 设置为每个批次都打印一次信息

        # print("print_freq", print_freq)

        start_time = time.time()#记录训练开始的时间。
        dataset_size = len(self.trainLoader.dataset)#获取训练集的大小。
        epoch_iter = 0#初始化epoch迭代次数为0。

        for batch, (inputs, labels) in enumerate(self.trainLoader):#遍历训练数据集

            inputs, labels = inputs.to(self.device), labels.to(self.device)#将输入数据和标签移动到设备上。
            # print("inputs", inputs.shape)
            epoch_iter += self.train_batch_size#更新每个epoch的迭代次数。

            ensemble_output = 0.0#初始化集成模型的输出为0。
            outputs = []#存储各个普通模型的输出。
            total_feature_maps = []#存储各个模型的特征图。
            fusion_module_inputs = []#存储融合模块的输入。
            leader_output, leader_trans_fusion_output = self.leader_model(inputs)#获取领导模型的输出。
            # print("leader_trans_fusion_output", leader_trans_fusion_output.shape)
            for i in range(self.model_num):#循环遍历普通模型
                outputs.append(self.models[i](inputs))#将当前普通模型的输出添加到列表中。
                ensemble_output += outputs[-1]#累加普通模型的输出，用于后续的集成。
                total_feature_maps.append(list(self.models[i].total_feature_maps.values()))#当前普通模型的特征图添加到列表中。
                fusion_module_inputs.append(list(self.models[i].total_feature_maps.values())[-1].detach())#将当前普通模型的最后一个特征图添加到融合模块的输入列表中。
            fusion_module_inputs = torch.cat(fusion_module_inputs, dim=1)#将融合模块的输入列表拼接成一个张量。
            # print("fusion_module_inputs", fusion_module_inputs.shape)
            fusion_module_inputs = F.interpolate(fusion_module_inputs, size=leader_trans_fusion_output.shape[2:],
                                                 mode='bilinear', align_corners=False)
            fusion_output = self.fusion_module(fusion_module_inputs)#通过融合模块获得融合后的输出。
            ensemble_output = ensemble_output / self.model_num

            # backward models   对每个普通模型和融合模块进行反向传播的过程，计算损失并更新模型参数。
            for i in range(self.model_num):

                loss_ce = self.criterion_CE(outputs[i], labels)
                loss_dml = 0.0

                for j in range(self.model_num):
                    if i != j:
                        loss_dml += self.criterion_KL(F.log_softmax(outputs[i] / self.temperature, dim=1),
                                                      F.softmax(outputs[j].detach() / self.temperature, dim=1))

                if i != 0:
                    current_attention_map = total_feature_maps[i][-1].pow(2).mean(1, keepdim=True)
                    other_attention_map = total_feature_maps[i - 1][-1].detach().pow(2).mean(1, keepdim=True)
                    loss_diversity = self.lambda_diversity * self.diversity_loss(current_attention_map,
                                                                                 other_attention_map)
                    loss_self_distllation = self.lambda_diversity * \
                                            self.self_distillation_loss(self.sd_models[i - 1],
                                                                        total_feature_maps[i],
                                                                        input_feature_map=self.diversity_target(
                                                                            total_feature_maps[i - 1][-1].detach()))
                else:
                    loss_diversity = 0.0
                    loss_self_distllation = 0.0
                # 当model_num=1时，没有其他模型可以相互学习，loss_dml应该为0
                if self.model_num > 1:
                    loss_dml = (self.temperature ** 2) * loss_dml / (self.model_num - 1)
                else:
                    loss_dml = 0.0
                loss = loss_ce + loss_dml + loss_diversity + loss_self_distllation

                # measure accuracy and record loss
                prec = utils.accuracy(outputs[i].data, labels.data, topk=topk)
                losses[i].update(loss.item(), inputs.size(0))
                ce_losses[i].update(loss_ce.item(), inputs.size(0))
                dml_losses[i].update(loss_dml, inputs.size(0))
                diversity_losses[i].update(loss_diversity, inputs.size(0))
                self_distillation_losses[i].update(loss_self_distllation, inputs.size(0))
                accuracy[i].update(prec[0], inputs.size(0))

                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()

            # backward fusion module
            loss_fusion_ce = self.criterion_CE(fusion_output, labels)#计算融合模块的交叉熵损失。
            loss_fusion_ensemble = (self.temperature ** 2) * self.criterion_KL(
                F.log_softmax(fusion_output / self.temperature, dim=1),
                F.softmax(ensemble_output.detach() / self.temperature, dim=1))#融合模块的集成损失。
            loss_fusion = loss_fusion_ce + loss_fusion_ensemble#融合模块的总损失。
            self.fusion_optimizer.zero_grad()#将普通模型的梯度清零。
            loss_fusion.backward()#反向传播计算梯度。
            self.fusion_optimizer.step()#更新普通模型的参数。

            fusion_ce_loss.update(loss_fusion_ce.item(), inputs.size(0))#更新融合模块的交叉熵损失。
            fusion_ensemble_loss.update(loss_fusion_ensemble.item(), inputs.size(0))#更新融合模块的集成损失。
            fusion_loss.update(loss_fusion.item(), inputs.size(0))#更新融合模块的总损失
            fusion_prec = utils.accuracy(fusion_output, labels.data, topk=topk)#计算融合模块的准确率。
            fusion_accuracy.update(fusion_prec[0], inputs.size(0))#更新融合模块的准确率。

            # backward leader model  对leader模型处理
            leader_feature_maps = list(self.leader_model.total_feature_maps.values())#取领导模型的所有特征图。
            loss_leader_ce = self.criterion_CE(leader_output, labels)#算领导模型的交叉熵损失。
            loss_leader_ensemble = (self.temperature ** 2) * self.criterion_KL(
                F.log_softmax(leader_output / self.temperature, dim=1),
                F.softmax(fusion_output.detach() / self.temperature, dim=1))#计算领导模型的集成损失，用于保证领导模型和融合模块的输出一致性。
            loss_leader_fusion = self.lambda_fusion * self.fusion_loss(#计算领导模型的融合损失
                leader_feature_maps[-1].pow(2).mean(1, keepdim=True),
                list(self.fusion_module.total_feature_maps.values())[-1].detach().pow(2).mean(1, keepdim=True))
           # 计算领导模型的融合转换损失
            loss_leader_trans_fusion = self.lambda_fusion * \
                                       self.fusion_loss(leader_trans_fusion_output.pow(2).mean(1, keepdim=True),
                                                           fusion_module_inputs.pow(2).mean(1, keepdim=True))
           #计算领导模型的自蒸馏损失
            loss_leader_self_distillation = self.lambda_fusion * \
                                            self.self_distillation_loss(self.sd_leader_model, leader_feature_maps,
                                                                        input_feature_map=list(
                                                                            self.fusion_module.total_feature_maps.values())[
                                                                            -1].detach())
           #计算领导模型的总损失
            loss_leader = loss_leader_ce + loss_leader_ensemble + loss_leader_fusion + loss_leader_trans_fusion + loss_leader_self_distillation

            self.leader_optimizer.zero_grad()#清空领导模型的梯度
            loss_leader.backward()#反向传播计算梯度
            self.leader_optimizer.step()#更新领导模型的参数。
#更新领导模型的损失和准确率记录器。
            leader_ce_loss.update(loss_leader_ce.item(), inputs.size(0))
            leader_ensemble_loss.update(loss_leader_ensemble.item(), inputs.size(0))
            leader_fusion_loss.update(loss_leader_fusion, inputs.size(0))
            leader_trans_fusion_loss.update(loss_leader_trans_fusion, inputs.size(0))
            leader_self_distillation_loss.update(loss_leader_self_distillation, inputs.size(0))
            leader_loss.update(loss_leader.item(), inputs.size(0))
            leader_prec = utils.accuracy(leader_output, labels.data, topk=topk)
            leader_accuracy.update(leader_prec[0], inputs.size(0))

            # update self distillation model after all models updated
            for i in range(1, self.model_num):
                # 对每个模型进行自蒸馏的计算
                loss_self_distillation_feature, loss_self_distillation_attention = \
                    self.train_self_distillation_model(self.sd_models[i - 1],
                                                       self.sd_optimizers[i - 1],
                                                       target_feature_maps=total_feature_maps[i])
                # 更新对应的损失计算器
                self_distillation_feature_losses[i].update(loss_self_distillation_feature, inputs.size(0))
                self_distillation_attention_losses[i].update(loss_self_distillation_attention, inputs.size(0))
            # 对领导模型进行自蒸馏的计算
            loss_leader_self_distillation_feature, loss_leader_self_distillation_attention = \
                self.train_self_distillation_model(self.sd_leader_model,
                                                   self.sd_leader_optimizer,
                                                   target_feature_maps=leader_feature_maps)
            # 更新对应的损失计算器
            leader_self_distillation_feature_loss.update(loss_leader_self_distillation_feature, inputs.size(0))
            leader_self_distillation_attention_loss.update(loss_leader_self_distillation_attention, inputs.size(0))
            # 计算平均准确率和集成准确率
            average_prec = utils.average_accuracy(outputs, labels.data, topk=topk)
            ensemble_prec = utils.ensemble_accuracy(outputs, labels.data, topk=topk)
            # 更新对应的指标计算器
            average_accuracy.update(average_prec[0], inputs.size(0))
            ensemble_accuracy.update(ensemble_prec[0], inputs.size(0))

            # 控制训练过程中信息输出的频率
            if batch % print_freq == 0 and batch != 0:
                current_time = time.time()
                cost_time = current_time - start_time

                msg = 'Epoch[{}] ({}/{})\tTime {:.2f}s\t'.format(
                    epoch, batch * self.train_batch_size, len(self.trainLoader.dataset), cost_time)
                total_losses = OrderedDict()
                total_losses['Fusion CE'] = float(fusion_ce_loss.avg)
                total_losses['Fusion Ensemble'] = float(fusion_ensemble_loss.avg)
                total_losses['Fusion Accuracy'] = float(fusion_accuracy.avg)
                total_losses['Fusion Loss'] = float(fusion_loss.avg)

                total_losses['Leader CE'] = float(leader_ce_loss.avg)
                total_losses['Leader Ensemble'] = float(leader_ensemble_loss.avg)
                total_losses['Leader Accuracy'] = float(leader_accuracy.avg)
                total_losses['Leader Self Distillation Feature'] = float(leader_self_distillation_feature_loss.avg)
                total_losses['Leader Self Distillation Attention'] = float(leader_self_distillation_attention_loss.avg)
                total_losses['Leader Self Distillation'] = float(leader_self_distillation_loss.avg)
                total_losses['Leader Fusion'] = float(leader_fusion_loss.avg)
                total_losses['Leader Trans Fusion'] = float(leader_trans_fusion_loss.avg)
                total_losses['Leader Loss'] = float(leader_loss.avg)
                for i in range(self.model_num):
                    total_losses['CE%d' % i] = float(ce_losses[i].avg)
                    total_losses['DML%d' % i] = float(dml_losses[i].avg)
                    total_losses['Diversity%d' % i] = float(diversity_losses[i].avg)
                    total_losses['Self Distillation Feature%d' % i] = float(self_distillation_feature_losses[i].avg)
                    total_losses['Self Distillation Attention%d' % i] = float(self_distillation_attention_losses[i].avg)
                    total_losses['Self Distillation%d' % i] = float(self_distillation_losses[i].avg)
                    total_losses['Loss%d' % i] = float(losses[i].avg)

                    msg += '|Model[{}]: Loss:{:.4f}\t' \
                           'CE Loss:{:.4f}\tDML Loss:{:.4f}\t' \
                           'Diversity Loss:{:.4f}\tSD Feature:{:.4f}' \
                           'SD Attention:{:.4f}\tSelf Distillation Loss:{:.4f}\t' \
                           'Accuracy {:.2f}%\t'.format(
                        i, float(losses[i].avg), float(ce_losses[i].avg), float(dml_losses[i].avg),
                        float(diversity_losses[i].avg), float(self_distillation_feature_losses[i].avg),
                        float(self_distillation_attention_losses[i].avg), float(self_distillation_losses[i].avg),
                        float(accuracy[i].avg))
                msg += '|Model[{}]: Loss:{:.4f}\t' \
                       'CE Loss:{:.4f}\tKL Loss:{:.4f}\t' \
                       'Accuracy {:.2f}%\t'.format(
                    'fusion', float(fusion_loss.avg), float(fusion_ce_loss.avg), float(fusion_ensemble_loss.avg),
                    float(fusion_accuracy.avg))
                msg += '|Model[{}]: Loss:{:.4f}\t' \
                       'CE Loss:{:.4f}\tEnsemble Loss:{:.4f}\t' \
                       'Fusion Loss:{:.4f}\tTrans Fusion Loss:{:.4f}\t' \
                       'SD Feature:{:.4f}\tSD Attention:{:.4f}\t' \
                       'Self Distillation Loss:{:.4f}\tAccuracy {:.2f}%\t'.format(
                    'leader', float(leader_loss.avg), float(leader_ce_loss.avg),
                    float(leader_ensemble_loss.avg), float(leader_fusion_loss.avg), float(leader_trans_fusion_loss.avg),
                    float(leader_self_distillation_feature_loss.avg),
                    float(leader_self_distillation_attention_loss.avg),
                    float(leader_self_distillation_loss.avg), float(leader_accuracy.avg))

                if self.opt.display_id > 0:
                    self.visualizer.plot_current_losses(epoch,
                                                        float(epoch_iter / (dataset_size * self.train_batch_size)),
                                                        total_losses)
                msg += '|Average Acc:{:.2f}\tEnsemble Acc:{:.2f}'.format(float(average_accuracy.avg),
                                                                         float(ensemble_accuracy.avg))
                self.logger.info(msg)

                start_time = current_time

    def test(self, epoch, topk=(1,)):

        losses = []
        accuracy = []
        top5_accuracy = []
        fusion_accuracy = utils.AverageMeter()
        leader_accuracy = utils.AverageMeter()
        average_accuracy = utils.AverageMeter()
        ensemble_accuracy = utils.AverageMeter()
        self.fusion_module.eval()#将融合模型设置为评估模式
        self.leader_model.eval()#将领导模型设置为评估模式，同样会影响模型中的某些层的行为。
        for i in range(self.model_num):
            self.models[i].eval()#将当前模型设置为评估模式。
            accuracy.append(utils.AverageMeter())#将一个新的用于计算平均值的工具类实例添加到准确率列表中。
            top5_accuracy.append(utils.AverageMeter())#工具类实例添加到top-5准确率列表中。
        accuracy.append(fusion_accuracy)
        accuracy.append(leader_accuracy)


        ################################################################

        all_labels = []
        all_outputs = [[] for _ in range(self.model_num)]
        all_fusion_outputs = []
        all_leader_outputs = []
        all_leader_predictions = []

        start_time = time.time()#记录当前时间，用于计算评估过程的持续时间。
        with torch.no_grad():#进入一个不需要计算梯度的上下文管理器，因为在评估阶段不需要计算梯度。
            for batch_idx, (inputs, labels) in enumerate(self.testLoader):#遍历测试数据加载器的每个批次。
                inputs, labels = inputs.to(self.device), labels.to(self.device)#将输入和标签移动到指定的设备（通常是GPU）上。

                outputs = []
                fusion_module_inputs = []#用于存储每个模型的特征图，以用于融合模型的输入。
                leader_output, _ = self.leader_model(inputs)#获取领导模型的输出。
                # print("leader_output", leader_output)
                for i in range(self.model_num):
                    outputs.append(self.models[i](inputs))# 计算当前模型的输出并将其添加到输出列表中。
                    #将当前模型的最后一个特征图添加到融合模型输入列表中，同时分离（detach）它以避免梯度计算。
                    fusion_module_inputs.append(list(self.models[i].total_feature_maps.values())[-1].detach())
                fusion_module_inputs = torch.cat(fusion_module_inputs, dim=1)# 将所有模型的特征图连接起来，以作为融合模型的输入。
                fusion_output = self.fusion_module(fusion_module_inputs)#计算融合模型的输出。

                # measure accuracy and record loss
                for i in range(self.model_num):
                    prec = utils.accuracy(outputs[i].data, labels.data, topk=topk)
                    # print("prec", prec)
                    accuracy[i].update(prec[0], inputs.size(0))
                    if len(topk) == 2:#检查是否计算了top-5准确率。
                        top5_accuracy[i].update(prec[1], inputs.size(0))#更新相应的计量器。

                fusion_prec = utils.accuracy(fusion_output, labels.data, topk=topk)
                fusion_accuracy.update(fusion_prec[0], inputs.size(0))

                leader_prec = utils.accuracy(leader_output, labels.data, topk=topk)
                leader_accuracy.update(leader_prec[0], inputs.size(0))

                average_prec = utils.average_accuracy(outputs, labels.data, topk=topk)
                ensemble_prec = utils.ensemble_accuracy(outputs, labels.data, topk=topk)

                average_accuracy.update(average_prec[0], inputs.size(0))
                ensemble_accuracy.update(ensemble_prec[0], inputs.size(0))
########################################################



                all_labels.append(labels.cpu().numpy())
                for i in range(self.model_num):
                    all_outputs[i].append(outputs[i].cpu().numpy())
                all_fusion_outputs.append(fusion_output.cpu().numpy())
                all_leader_outputs.append(leader_output.cpu().numpy())
                all_leader_predictions.append(np.argmax(leader_output.cpu().numpy(), axis=1))


            all_labels = np.concatenate(all_labels)
            all_outputs = [np.concatenate(output) for output in all_outputs]
            all_fusion_outputs = np.concatenate(all_fusion_outputs)
            all_leader_outputs = np.concatenate(all_leader_outputs)
            all_leader_predictions = np.concatenate(all_leader_predictions)

            auc_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            specificity_scores = []

            # 在循环中计算每个模型的特异性
            for i in range(self.model_num):
                tn, fp, _, _ = confusion_matrix(all_labels, np.argmax(all_outputs[i], axis=1)).ravel()
                specificity = tn / (tn + fp) if tn + fp != 0 else 0.0  # 避免除零错误
                specificity_scores.append(specificity)

            # 计算融合模型和领导模型的特异性
            fusion_tn, fusion_fp, _, _ = confusion_matrix(all_labels, np.argmax(all_fusion_outputs, axis=1)).ravel()
            fusion_specificity = fusion_tn / (fusion_tn + fusion_fp) if fusion_tn + fusion_fp != 0 else 0.0
            leader_tn, leader_fp, _, _ = confusion_matrix(all_labels, np.argmax(all_leader_outputs, axis=1)).ravel()
            leader_specificity = leader_tn / (leader_tn + leader_fp) if leader_tn + leader_fp != 0 else 0.0

            for i in range(self.model_num):
                auc_scores.append(roc_auc_score(all_labels, all_outputs[i][:, 1]))
                precision_scores.append(
                    precision_score(all_labels, np.argmax(all_outputs[i], axis=1), average='macro', zero_division=0))
                recall_scores.append(
                    recall_score(all_labels, np.argmax(all_outputs[i], axis=1), average='macro', zero_division=0))
                f1_scores.append(
                    f1_score(all_labels, np.argmax(all_outputs[i], axis=1), average='macro', zero_division=0))


            fusion_auc = roc_auc_score(all_labels, all_fusion_outputs[:, 1])
            leader_auc = roc_auc_score(all_labels, all_leader_outputs[:, 1])

            fusion_precision = precision_score(all_labels, np.argmax(all_fusion_outputs, axis=1), average='macro',
                                               zero_division=0)
            leader_precision = precision_score(all_labels, np.argmax(all_leader_outputs, axis=1), average='macro',
                                               zero_division=0)

            fusion_recall = recall_score(all_labels, np.argmax(all_fusion_outputs, axis=1), average='macro',
                                         zero_division=0)
            leader_recall = recall_score(all_labels, np.argmax(all_leader_outputs, axis=1), average='macro',
                                         zero_division=0)

            fusion_f1 = f1_score(all_labels, np.argmax(all_fusion_outputs, axis=1), average='macro', zero_division=0)
            leader_f1 = f1_score(all_labels, np.argmax(all_leader_outputs, axis=1), average='macro', zero_division=0)


            current_time = time.time()#记录评估过程结束时的时间。
#构建包含当前时期和评估时间的消息字符串。
            msg = 'Epoch[{}]\tTime {:.2f}s\t'.format(epoch, current_time - start_time)
            for i in range(self.model_num):
                msg += 'Model[{}]:\tAccuracy {:.2f}%\tAUC {:.2f}\tPrecision {:.2f}\tSensitivity {:.2f}\tSpecificity {:.2f}%\tF1 {:.2f}\t'.format(
                    i, float(accuracy[i].avg), auc_scores[i] * 100, precision_scores[i] * 100, recall_scores[i] * 100, specificity_scores[i] * 100,
                                                f1_scores[i] * 100)
            msg += 'Model[{}]:\tAccuracy {:.2f}%\tAUC {:.2f}\tPrecision {:.2f}\tSensitivity {:.2f}\tSpecificity {:.2f}%\tF1 {:.2f}\t'.format(
                'Fusion', float(fusion_accuracy.avg), fusion_auc * 100, fusion_precision * 100, fusion_recall * 100, fusion_specificity * 100,
                                                     fusion_f1 * 100)
            msg += 'Model[{}]:\tAccuracy {:.2f}%\tAUC {:.2f}\tPrecision {:.2f}\tSensitivity {:.2f}\tSpecificity {:.2f}%\tF1 {:.2f}\t'.format(
                'Leader', float(leader_accuracy.avg), leader_auc * 100, leader_precision * 100, leader_recall * 100, leader_specificity * 100,
                                                      leader_f1 * 100)

            msg += 'Average Acc:{:.2f}\tEnsemble Acc:{:.2f}'.format(float(average_accuracy.avg),
                                                                    float(ensemble_accuracy.avg))
            # 输出预测值和真实标签
            msg += '\nPredictions: {}\nLabels: {}\n'.format(np.argmax(all_leader_outputs, axis=1), all_labels)
            self.logger.info(msg + '\n')

        return losses, accuracy, top5_accuracy, average_accuracy, ensemble_accuracy








    def train_self_distillation_model(self, sd_model, sd_optimizer, target_feature_maps):

        sd_model.train()
        sd_feature_loss = 0.0#初始化
        sd_attention_loss = 0.0
        input = target_feature_maps[-1].detach()#从目标特征图中获取最后一个特征图，并分离它以防止梯度传播。
        sd_model(input)#使用最后一个特征图作为输入，前向传播自我蒸馏模型。
        total_feature_maps = list(sd_model.total_feature_maps.values())#获取自我蒸馏模型的所有特征图。
        total_feature_maps.reverse()#反转特征图的顺序，以便从最深层开始计算损失。

        for i, feature_map in enumerate(total_feature_maps):
            attention_map = feature_map.pow(2).mean(1, keepdim=True)#计算当前特征图的注意力图。
            target_attenion_map = target_feature_maps[i].detach().pow(2).mean(1, keepdim=True)#获取目标特征图的注意力图。

            #计算特征损失，将其加到总特征损失中。
            sd_feature_loss += self.lambda_self_distillation * \
                               self.attention_loss(feature_map,
                                                   target_feature_maps[i].detach())
           #计算注意力损失，将其加到总注意力损失中。
            sd_attention_loss += self.lambda_self_distillation * \
                                 self.attention_loss(attention_map,
                                                     target_attenion_map)

        sd_loss = sd_feature_loss + sd_attention_loss

        sd_optimizer.zero_grad()
        sd_loss.backward()# 反向传播总损失
        sd_optimizer.step()

        return sd_feature_loss, sd_attention_loss# 返回特征损失和注意力损失。

    def self_distillation_loss(self, sd_model, source_feature_maps, input_feature_map=None):

        sd_model.eval()#将自我蒸馏模型设置为评估模式，这意味着模型不会更新参数
        sd_loss = 0.0

        if input_feature_map is None:
            input_feature_map = source_feature_maps[-1].detach()#如果输入特征图为None，则将其设置为源特征图的最后一个特征图，并分离它以防止梯度传播。
        else:
            input_feature_map = input_feature_map.detach()#分离输入特征图以防止梯度传播。
        sd_model(input_feature_map)#- 使用输入特征图前向传播自我蒸馏模型。
        target_feature_maps = list(sd_model.total_feature_maps.values())#获取自我蒸馏模型的所有特征图。
        target_feature_maps.reverse()#反转目标特征图的顺序，以便从最深层开始计算损失。

        for i, feature_map in enumerate(target_feature_maps):
            source_attention_map = source_feature_maps[i].pow(2).mean(1, keepdim=True)

            target_attention_map = feature_map.detach().pow(2).mean(1, keepdim=True)


            if source_attention_map.shape != target_attention_map.shape:
                # print("source_attention_map shape:", source_attention_map.shape)
                # print("target_attention_map shape before interpolation:", target_attention_map.shape)
                target_attention_map = F.interpolate(target_attention_map, size=source_attention_map.shape[2:],
                                                     mode='bilinear', align_corners=False)

            #
            # print("source_attention_map", source_attention_map.shape)
            #
            # print("target_attention_map", target_attention_map.shape)
            # print("target_attention_map shape after interpolation:", target_attention_map.shape)

            sd_loss += self.attention_loss(source_attention_map, target_attention_map)#算并累加自我蒸馏损失，该损失衡量源和目标特征图的注意力图之间的差异。
            # print("sd_loss", sd_loss)

        return sd_loss#返回自我蒸馏损失。


    def lambda_warmup(self, epoch):#温和，提高模型在训练初期的效率和稳定性，从而加快模型收敛的速度并提高其性能。

        def warmup(lambda_coeff, epoch, alpha=5):

            if epoch <= alpha:
                return lambda_coeff * math.exp(-5 * math.pow((1 - float(epoch) / alpha), 2))
            else:
                return lambda_coeff

        self.lambda_diversity = warmup(self.opt.lambda_diversity, epoch)
        self.lambda_fusion = warmup(self.opt.lambda_fusion, epoch)
        # print("lambda_fusion", self.lambda_fusion.shape)
        self.lambda_self_distillation = warmup(self.opt.lambda_self_distillation, epoch)

    #在模型训练过程中引入一些差异性，以帮助模型更好地探索数据的特征空间，从而提高模型的泛化能力。
    def diversity_target(self, y):
        # print("y.size", y.shape)
        attention_y = y.pow(2).mean(1, keepdim=True)
        attention_y = attention_y * 1e7
        # print("attention_y", attention_y)
        attention_y_size = attention_y.size()
        # print("attention_y_size", attention_y_size)

        norm_y = torch.norm(attention_y.view(attention_y.size(0), -1), dim=1, keepdim=True)
        # print("norm_y", norm_y)

        attention_y = F.normalize(attention_y.view(attention_y.size(0), -1))

        # threshold = attention_y.topk(1, largest=False)[0][:, -1].unsqueeze(-1)
        threshold = attention_y.topk(int(attention_y.size(1) / 3), largest=False)[0][:, -1].unsqueeze(-1)

        # print("threshold_11", threshold)

        # 修改这里的 attention_y_size
        # attention_y_size = torch.Size([16, 1, 16, 1])

        target_y = (norm_y / 2 - attention_y) * torch.sign(attention_y - threshold) + norm_y / 2
        diff = (target_y - attention_y.view(attention_y.size(0), -1))
        # print("(diff * norm_y / y.size(0))", (diff * norm_y/ y.size(0)).shape)
        return y + ((diff * norm_y / y.size(0)).view(attention_y_size))
# L2 损失函数
    def l2_loss(self, source_feature_maps, target_feature_maps):

        l2_loss = 0.0
        for i, feature_map in enumerate(source_feature_maps):
            l2_loss += self.attention_loss(feature_map.pow(2).mean(1, keepdim=True),
                                           target_feature_maps[i].detach().pow(2).mean(1, keepdim=True))
        return l2_loss
#多样性损失函数
    def diversity_loss(self, x, y, scale_factor_x=1e7, scale_factor_y=1e7):
        # 打印归一化之前的数据
        # 放大数据
        # x = x * scale_factor_x
        # y = y * scale_factor_y
        # print("x before normalization:", x.view(x.size(0), -1))
        # print("y before normalization:", y.view(y.size(0), -1))

        norm_y = torch.norm(y.view(y.size(0), -1), dim=1, keepdim=True)
        # print("norm_y", norm_y.shape)
        x = F.normalize(x.view(x.size(0), -1))
        # print("x", x.shape)
        y = F.normalize(y.view(y.size(0), -1))
        # print("y", y.shape)
        # y_t = y.size(0)
        # print("y_t", y_t)
        # y_tst = y.size(1)/3
        # print("y_tst", y_tst)

        y_test = y.topk(int(y.size(1)/3), largest=False)
        # print("y_test", y_test)
        threshold = y.topk(int(y.size(1) / 3), largest=False)[0][:, -1].unsqueeze(-1)

        # print("threshold", threshold.shape)
        y = (norm_y / 2 - y) * torch.sign(y - threshold) + norm_y / 2
        # print("y", y)
        return (x - y).pow(2).mean()








    def fusion_loss(self, x, y):

        x = F.normalize(x.view(x.size(0), -1))
        y = F.normalize(y.view(y.size(0), -1))
        # if x.size() != y.size():
        #     # 假设需要插值来调整尺寸
        #     y = torch.nn.functional.interpolate(y, size=x.size()[2:])
        if x.size() != y.size():
            # 假设输入张量是 [batch_size, features]
            if len(y.size()) == 2:
                y = y.unsqueeze(1).unsqueeze(-1)  # [batch_size, features] -> [batch_size, 1, features, 1]
                x = x.unsqueeze(1).unsqueeze(-1)  # [batch_size, features] -> [batch_size, 1, features, 1]

            y = torch.nn.functional.interpolate(y, size=x.size()[2:], mode='nearest')  # 调整尺寸

            # 去掉添加的维度
            y = y.squeeze(1).squeeze(-1)  # [batch_size, 1, features, 1] -> [batch_size, features]
            x = x.squeeze(1).squeeze(-1)  # [batch_size, 1, features, 1] -> [batch_size, features]
        return (x - y).pow(2).mean()






    def attention_loss(self, x, y):
        #
        # x = F.normalize(x.view(x.size(0), -1))
        # y = F.normalize(y.view(y.size(0), -1))
        x = F.normalize(x.view(x.size(0), -1), p=2, dim=1)
        y = F.normalize(y.view(y.size(0), -1), p=2, dim=1)
        # 使用插值将y的形状调整为与x相同
        y = F.interpolate(y.unsqueeze(1), size=x.shape[-1], mode='nearest').squeeze(1)
        # print("x", x.shape)
        # print("y", y.shape)







        return (x - y).pow(2).mean()
#加载模型的权重参数
    def load_models(self, model, opt):

        if opt.load_path is None or not os.path.exists(opt.load_path):
            raise FileExistsError('Load path must be exist!!!')
        ckpt = torch.load(opt.load_path, map_location=self.device)
        model.load_state_dict(ckpt['weight'])





    def save_models(self, model, epoch, name, opt, isbest):

        save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints')
        utils.mkdirs(save_dir)
        ckpt = {
            'weight': model.state_dict(),
            'epoch': epoch,
            'cfg': opt.model,
            'index': name
        }
        if isbest:
            torch.save(ckpt, os.path.join(save_dir, 'model%s_best.pth' % name))
        else:
            torch.save(ckpt, os.path.join(save_dir, 'model%s_%d.pth' % (name, epoch)))




#     def train_self_distillation_model(self, sd_model, sd_optimizer, target_feature_maps):
#
#         sd_model.train()
#         sd_feature_loss = 0.0#初始化
#         sd_attention_loss = 0.0
#         input = target_feature_maps[-1].detach()#从目标特征图中获取最后一个特征图，并分离它以防止梯度传播。
#         sd_model(input)#使用最后一个特征图作为输入，前向传播自我蒸馏模型。
#         total_feature_maps = list(sd_model.total_feature_maps.values())#获取自我蒸馏模型的所有特征图。
#         total_feature_maps.reverse()#反转特征图的顺序，以便从最深层开始计算损失。
#
#         for i, feature_map in enumerate(total_feature_maps):
#             attention_map = feature_map.pow(2).mean(1, keepdim=True)#计算当前特征图的注意力图。
#             target_attenion_map = target_feature_maps[i].detach().pow(2).mean(1, keepdim=True)#获取目标特征图的注意力图。
# #计算特征损失，将其加到总特征损失中。
#             sd_feature_loss += self.lambda_self_distillation * \
#                                self.attention_loss(feature_map,
#                                                    target_feature_maps[i].detach())
#            #计算注意力损失，将其加到总注意力损失中。
#             sd_attention_loss += self.lambda_self_distillation * \
#                                  self.attention_loss(attention_map,
#                                                      target_attenion_map)
#
#         sd_loss = sd_feature_loss + sd_attention_loss
#
#         sd_optimizer.zero_grad()
#         sd_loss.backward()# 反向传播总损失0
#         sd_optimizer.step()
#
#         return sd_feature_loss, sd_attention_loss# 返回特征损失和注意力损失。
#
#     def self_distillation_loss(self, sd_model, source_feature_maps, input_feature_map=None):
#
#         sd_model.eval()#将自我蒸馏模型设置为评估模式，这意味着模型不会更新参数
#         sd_loss = 0.0
#
#         if input_feature_map is None:
#             input_feature_map = source_feature_maps[-1].detach()#如果输入特征图为None，则将其设置为源特征图的最后一个特征图，并分离它以防止梯度传播。
#         else:
#             input_feature_map = input_feature_map.detach()#分离输入特征图以防止梯度传播。
#         sd_model(input_feature_map)#- 使用输入特征图前向传播自我蒸馏模型。
#         target_feature_maps = list(sd_model.total_feature_maps.values())#获取自我蒸馏模型的所有特征图。
#         target_feature_maps.reverse()#反转目标特征图的顺序，以便从最深层开始计算损失。
#
#         for i, feature_map in enumerate(target_feature_maps):
#             source_attention_map = source_feature_maps[i].pow(2).mean(1, keepdim=True)
#             target_attention_map = feature_map.detach().pow(2).mean(1, keepdim=True)
#             sd_loss += self.attention_loss(source_attention_map, target_attention_map)#算并累加自我蒸馏损失，该损失衡量源和目标特征图的注意力图之间的差异。
#
#         return sd_loss#返回自我蒸馏损失。
#
#     def lambda_warmup(self, epoch):#温和，提高模型在训练初期的效率和稳定性，从而加快模型收敛的速度并提高其性能。
#
#         def warmup(lambda_coeff, epoch, alpha=5):
#
#             if epoch <= alpha:
#                 return lambda_coeff * math.exp(-5 * math.pow((1 - float(epoch) / alpha), 2))
#             else:
#                 return lambda_coeff
#
#         self.lambda_diversity = warmup(self.opt.lambda_diversity, epoch)
#         self.lambda_fusion = warmup(self.opt.lambda_fusion, epoch)
#         self.lambda_self_distillation = warmup(self.opt.lambda_self_distillation, epoch)
#
#     #在模型训练过程中引入一些差异性，以帮助模型更好地探索数据的特征空间，从而提高模型的泛化能力。
#     def diversity_target(self, y):
#
#         attention_y = y.pow(2).mean(1, keepdim=True)
#         attention_y_size = attention_y.size()
#         norm_y = torch.norm(attention_y.view(attention_y.size(0), -1), dim=1, keepdim=True)
#         attention_y = F.normalize(attention_y.view(attention_y.size(0), -1))
#         threshold = attention_y.topk(int(attention_y.size(1) / 3), largest=False)[0][:, -1].unsqueeze(-1)
#         target_y = (norm_y / 2 - attention_y) * torch.sign(attention_y - threshold) + norm_y / 2
#         diff = (target_y - attention_y.view(attention_y.size(0), -1))
#         return y + ((diff * norm_y / y.size(1)).view(attention_y_size))
# # L2 损失函数
#     def l2_loss(self, source_feature_maps, target_feature_maps):
#
#         l2_loss = 0.0
#         for i, feature_map in enumerate(source_feature_maps):
#             l2_loss += self.attention_loss(feature_map.pow(2).mean(1, keepdim=True),
#                                            target_feature_maps[i].detach().pow(2).mean(1, keepdim=True))
#         return l2_loss
# #多样性损失函数
#     def diversity_loss(self, x, y):
#
#         norm_y = torch.norm(y.view(y.size(0), -1), dim=1, keepdim=True)
#         x = F.normalize(x.view(x.size(0), -1))
#         y = F.normalize(y.view(y.size(0), -1))
#         threshold = y.topk(int(y.size(1) / 3), largest=False)[0][:, -1].unsqueeze(-1)
#         y = (norm_y / 2 - y) * torch.sign(y - threshold) + norm_y / 2
#         return (x - y).pow(2).mean()
#
#     def fusion_loss(self, x, y):
#
#         x = F.normalize(x.view(x.size(0), -1))
#         y = F.normalize(y.view(y.size(0), -1))
#         return (x - y).pow(2).mean()
#
#     def attention_loss(self, x, y):
#
#         x = F.normalize(x.view(x.size(0), -1))
#         y = F.normalize(y.view(y.size(0), -1))
#         return (x - y).pow(2).mean()
# #加载模型的权重参数
#     def load_models(self, model, opt):
#
#         if opt.load_path is None or not os.path.exists(opt.load_path):
#             raise FileExistsError('Load path must be exist!!!')
#         ckpt = torch.load(opt.load_path, map_location=self.device)
#         model.load_state_dict(ckpt['weight'])
#
#     def save_models(self, model, epoch, name, opt, isbest):
#
#         save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints')
#         utils.mkdirs(save_dir)
#         ckpt = {
#             'weight': model.state_dict(),
#             'epoch': epoch,
#             'cfg': opt.model,
#             'index': name
#         }
#         if isbest:
#             torch.save(ckpt, os.path.join(save_dir, 'model%s_best.pth' % name))
#         else:
#             torch.save(ckpt, os.path.join(save_dir, 'model%s_%d.pth' % (name, epoch)))

# 作者：孙海滨
# 日期：2024/4/11
