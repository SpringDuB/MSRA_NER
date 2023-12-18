import logging
import os
import sys
sys.path.append('/mnt/workspace/nlp/MSRA_ner/')
# print(sys.path)
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.MSRA_ner.evaluate import evaluate_epoch
from src.MSRA_ner.models.model import build_model
from src.MSRA_ner.utils import logging_utils
from src.MSRA_ner.utils.dataloader import NERDataLoader
from src.MSRA_ner.utils.early_stop import EarlyStopStrategy
from src.MSRA_ner.utils.metrics import accuracy, ner_accuracy
from src.MSRA_ner.utils.optimizer import build_optimizer
from src.MSRA_ner.utils.param import Param, build_params
from src.MSRA_ner.utils.running_utils import RunningAverage


def train_epoch(epoch, train_step, model: nn.Module, data_loader, optimizer, param: Param, summary_writer,
                early_stopper: EarlyStopStrategy):
    """
    一个epoch的训练
    :param epoch: 当前epoch
    :param train_step: 当前初始的训练step
    :param model: 模型对象
    :param data_loader: 数据遍历器
    :param optimizer: 优化器
    :param param: 参数对象
    :param summary_writer: 日志输出对象
    :param early_stopper: 提前停止对象
    :return: train_step
    """
    device = param.device
    # 设置模型为train训练阶段
    model.to(device)
    model.train()

    # 遍历
    step = 1
    bar = tqdm(data_loader)
    loss_avg = RunningAverage()
    acc1_avg = RunningAverage()
    acc2_avg = RunningAverage()
    for input_ids, attention_mask, label_ids in bar:
        train_step += 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_ids = label_ids.to(device)

        # 计算损失
        scores, loss, pred_label_ids = model(input_ids, attention_mask, labels=label_ids, return_output=True)
        if param.n_gpu > 1:
            loss = loss.mean()  # 多gpu运行的时候模型返回的是一个tensor[N]结构
        # 梯度累加的时候，相当于损失平均
        if param.gradient_accumulation_steps > 1:
            loss = loss / param.gradient_accumulation_steps

        # 反向传播-求解梯度
        loss.backward()

        # 基于梯度进行参数更新
        if step % param.gradient_accumulation_steps == 0:
            optimizer.step()  # 参数更新
            optimizer.zero_grad()  # 将梯度重置为0

        # 日志信息描述
        acc1 = accuracy(pred_label_ids, label_ids, attention_mask)
        acc1_avg.update(acc1[1], acc1[2])
        acc2 = ner_accuracy(pred_label_ids, label_ids)
        acc2_avg.update(acc2[1], acc2[2])
        loss = loss.item() * param.gradient_accumulation_steps
        loss_avg.update(loss)
        bar.set_postfix(ordered_dict={
            'batch_loss': f'{loss:.3f}',
            'loss': f'{loss_avg():.3f}',
            'acc': f'{acc1_avg():.3f}/{acc1[0]:.3f}',
            'acc without O': f'{acc2_avg():.3f}/{acc2[0]:.3f}'
        })
        summary_writer.add_scalars(
            main_tag='train',
            tag_scalar_dict={
                'batch_loss': loss,
                'batch_acc_with_o': acc1[0],
                'batch_acc_without_o': acc2[0]
            },
            global_step=train_step
        )
        early_stopper.update(acc1[0])  # 使用全局的准确率
        if early_stopper.stop():
            break
    summary_writer.add_scalars(
        main_tag='train',
        tag_scalar_dict={
            'epoch_loss': loss_avg(),
            'epoch_acc_with_o': acc1_avg(),
            'epoch_acc_without_o': acc2_avg()
        },
        global_step=epoch
    )
    return train_step


def train_and_evaluate(model: nn.Module, optimizer, train_loader, val_loader, param: Param):
    # 日志输出对象的构建
    summary_writer = SummaryWriter(log_dir=param.summary_log_dir)
    # device = param.device
    # noinspection PyTypeChecker
    # summary_writer.add_graph(
    #     model,
    #     input_to_model=[torch.randint(10, (2, 20)).to(device), torch.randint(2, (2, 20)).to(device), torch.randint(10, (2, 20)).to(device)]
    # )

    train_early_stopper = EarlyStopStrategy(max_stop_counter=500)  # 连续100个batch的准确率没有提升，直接结束训练
    val_early_stopper = EarlyStopStrategy(max_stop_counter=2)  # 连续两个epoch，f1指标没有提升，直接结束训练

    # # 模型保存
    torch.save(
        {
            'param': model.state_dict(),
            'epoch': 0
        },
        os.path.join(param.model_save_dir, f"{0:06d}.pkl")
    )

    # 训练&验证
    train_step = 0
    for epoch in range(1, param.epoch_num + 1):
        logging.info("Epoch {}/{}".format(epoch, param.epoch_num))

        # Train model
        train_step = train_epoch(epoch, train_step, model, train_loader, optimizer, param, summary_writer,
                                 train_early_stopper)

        # Evaluate模型效果
        val_metrics = evaluate_epoch(model, val_loader, param, mark='Val', verbose=True)
        val_f1 = val_metrics['f1']  # 验证集的f1指标
        val_early_stopper.update(val_f1, model)

        # # 模型保存
        torch.save(
            {
                'param': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer,
                'best_f1': val_early_stopper.best_eval_value
            },
            os.path.join(param.model_save_dir, f"{epoch:06d}.pkl")
        )

        # 提前停止判断
        if val_early_stopper.stop():
            logging.info(f"Val Early stop model training:{epoch}. Best val f1:{val_early_stopper.best_eval_value:.3f}")
            break
        if train_early_stopper.stop():
            logging.info(
                f"Train Early stop model training:{epoch}. Best val f1:{val_early_stopper.best_eval_value:.3f}")
            break
        if epoch == param.epoch_num:
            logging.info(f"Best val f1:{val_early_stopper.best_eval_value:.3f}")
            break

    # 最后的最优模型保存
    torch.save(
        {
            'param': val_early_stopper.best_model_state_dict,
            'best_f1': val_early_stopper.best_eval_value
        },
        os.path.join(param.model_save_dir, f"best.pkl")
    )

    # 关闭日志输出对象
    summary_writer.close()


def run():
    logging_utils.set_logger(True, log_path='running.log')

    logging.info("Start build param....")
    param = build_params()
    logging.info(f"Params:\n{param.to_dict()}")

    # 构建模型
    logging.info("Start build train model....")
    model = build_model(param).to(param.device)
    logging.info(f"Model:\n{model}")
    if param.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 加载训练数据和验证数据
    dataloader = NERDataLoader(model.lm_model.tokenizer_class, param)
    train_loader = dataloader.get_dataloader("train")
    val_loader = dataloader.get_dataloader("test")

    # 构建优化器
    logging.info("Start build optimizer....")
    total_train_batch = len(train_loader) // param.gradient_accumulation_steps * param.epoch_num  # 训练中的总参数更新次数
    optimizer = build_optimizer(model, param, total_train_batch=total_train_batch)

    # 训练&评估
    logging.info(f"Starting training for {param.epoch_num} epochs.")
    train_and_evaluate(model, optimizer, train_loader, val_loader, param)
    logging.info("Completed training!")


if __name__ == '__main__':
    run()
