# trainer.py
import sys, os
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "train"))
import torch
import time
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import stratified_sampling
from losses import compute_cvl_loss, compute_iso_loss, compute_velocity_consistency_loss
from utils import scaled_output


class INNTrainer:
    def __init__(self, model, config, train_data, train_t):
        self.config = config  # 直接保存整个配置字典
        self.device = config['device']  # 从配置中获取device
        self.model = model.to(self.device)

        self.train_data_np = train_data.cpu().numpy()
        self.train_t_np = train_t.cpu().numpy()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['lr_scheduler_t_max'],
            eta_min=self.config['lr_scheduler_eta_min']
        )

        self.best_model_state = None
        self.min_val_loss = float('inf')

    def train(self):
        print("--- Starting Training ---")

        for epoch in range(1, self.config['epochs'] + 1):
            self.model.train()
            self.optimizer.zero_grad()
            # 1. 数据采样
            batch_data, batch_t = stratified_sampling(
                self.train_data_np, self.train_t_np, self.config['batch_size']
            )
            batch_data, batch_t = batch_data.to(self.device), batch_t.to(self.device)

            # 2. 前向传播
            z, _ = self.model(batch_data)
            z_min = z.min(dim=0, keepdim=True)[0] # 分离计算图
            z_max = z.max(dim=0, keepdim=True)[0] # 分离计算图
            z_scaled = scaled_output(z)

            # 3. 计算损失
            loss_cvl = compute_cvl_loss(
                t=batch_t, z=z_scaled, input_data=batch_data, inn_model=self.model,
                z_min=z_min, z_max=z_max, ot_reg=self.config['OT_REGULARIZATION'], num_gap=self.config['num_gap']
            )
            loss_iso = compute_iso_loss(batch_data, z_scaled, batch_t)

            total_loss = self.config['lambda_cvl'] * loss_cvl + self.config['lambda_iso'] * loss_iso
            print(f'Epoch:{epoch}, cvl_loss:{loss_cvl}, iso_loss:{loss_iso}')
            # 4. 验证逻辑
            if epoch > self.config['val_epoch_start']:
                self.validate()

            # 5. 反向传播和优化
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['grad_clip_max_norm'])
            self.optimizer.step()
            self.scheduler.step()


    def validate(self):
        self.model.eval()
        with torch.no_grad():
            # 使用全部训练数据进行验证
            full_z, _ = self.model(torch.from_numpy(self.train_data_np).float().to(self.device))
            full_z_scaled = scaled_output(full_z)
            val_loss = compute_velocity_consistency_loss(
                torch.from_numpy(self.train_t_np).float().to(self.device), full_z_scaled
            )
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"  New best model saved with validation loss: {self.min_val_loss:.4f}")