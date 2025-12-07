import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Callable, Optional, List, Tuple
import matplotlib.pyplot as plt
import os

import time
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from utils.config import cfg
from collections import defaultdict
from sktime.forecasting.base import BaseForecaster



class PinballLoss:
    def __init__(
            self,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.quantiles = torch.tensor(cfg.QUANTILES, dtype=torch.float32, device=device)
    
    def __call__(self, target_data: torch.Tensor, predict_data: torch.Tensor):
        error_data = target_data - predict_data 
        loss_data = torch.max(self.quantiles * error_data, (self.quantiles - 1) * error_data)
        return loss_data.sum()

class QuantileAggregatorTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable, # PinballLoss
        device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
        compile_model: bool = False, # torch.compile иногда ломается на einsum/embedding
        enable_amp: bool = False,     # AMP может быть нестабилен для квантилей (fp16)
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Специализированный тренер для UniversalQuantileAggregator.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = torch.amp.GradScaler(self.device, enabled=enable_amp)
        self.enable_amp = enable_amp
        
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': {}}
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float('inf')

        os.makedirs(checkpoint_dir, exist_ok=True)

        if compile_model:
            try:
                self.model = torch.compile(model)
                print("Model compiled with torch.compile()")
            except Exception as e:
                print(f"Warning: Could not compile model. {e}")

    def _process_batch(self, batch):
        """
        Распаковывает батч в зависимости от структуры данных.
        Ожидает от DataLoader: [base_preds, context, target] или [base_preds, target]
        """
        if len(batch) == 3:
            # base_preds, context (series_id или features), target
            base_preds, context, target = batch
            return base_preds.to(self.device), context.to(self.device), target.to(self.device)
        elif len(batch) == 2:
            # base_preds, target (для Global агрегации, где context не нужен)
            base_preds, target = batch
            return base_preds.to(self.device), None, target.to(self.device)
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        for batch in train_loader:
            base_preds, context, target = self._process_batch(batch)

            self.optimizer.zero_grad()

            # AMP context
            with torch.amp.autocast(self.device, enabled=self.enable_amp):
                # Forward pass агрегатора
                # Модель ожидает (base_preds, context_data)
                final_preds = self.model(base_preds, context)
                loss = self.criterion(final_preds, target)

            if self.enable_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / max(1, num_batches)

    def validate(
        self,
        val_loader: DataLoader,
        metrics: Optional[Dict[str, Callable]] = None
    ) -> Tuple[float, Dict[str, float]]:
        
        self.model.eval()
        val_loss = 0.0
        metric_values = {name: 0.0 for name in metrics} if metrics else {}
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                base_preds, context, target = self._process_batch(batch)

                with torch.amp.autocast(self.device, enabled=self.enable_amp):
                    final_preds = self.model(base_preds, context)
                    loss = self.criterion(final_preds, target)

                val_loss += loss.item()

                if metrics:
                    # Метрики обычно ожидают cpu numpy
                    preds_np = final_preds.detach()
                    target_np = target.detach()
                    for name, metric_fn in metrics.items():
                        metric_values[name] += metric_fn(preds_np, target_np).item()

        avg_loss = val_loss / max(1, num_batches)
        if metrics:
            for name in metric_values:
                metric_values[name] /= max(1, num_batches)

        return avg_loss, metric_values

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        metrics: Optional[Dict[str, Callable]] = None,
        early_stopping: int = 10,
        scheduler = None # Learning rate scheduler
    ):
        print(f"Starting training on {self.device}...")
        
        for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader, metrics)
            
            if scheduler:
                # Если ReduceLROnPlateau -> step(val_loss), иначе step()
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            epoch_time = time.time() - start_time
            
            # Логирование
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for k, v in val_metrics.items():
                if k not in self.history['metrics']: self.history['metrics'][k] = []
                self.history['metrics'][k].append(v)
            
            msg = (f"Ep {epoch} | T={epoch_time:.1f}s | "
                   f"Tr Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_metrics:
                msg += f" | {val_metrics}"
            # tqdm.write(msg) # Чтобы не ломать прогресс бар

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_aggregator.pth")
                no_improve = 0
            else:
                no_improve += 1
            
            if early_stopping and no_improve >= early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
                
        self.load_checkpoint("best_aggregator.pth") # Возвращаем лучшую модель
        return self.history

    def save_checkpoint(self, filename):
        torch.save({
            'model_state': self.model.state_dict(),
            'best_loss': self.best_val_loss
        }, os.path.join(self.checkpoint_dir, filename))

    def load_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            self.best_val_loss = ckpt['best_loss']
            print(f"Loaded best model from {path}")

    def plot_history(self):
        history = self.history
        plt.figure(figsize=(15, 5))
        
        # Loss
        plt.subplot(3, 1, 1)
        plt.plot(history['train_loss'], label='Train Pinball')
        plt.title('Train Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 2)
        plt.plot(history['val_loss'], color='orange', label='Val Pinball')
        plt.title('Val Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics
        if history['metrics']:
            plt.subplot(3, 1, 3)
            for name, values in history['metrics'].items():
                plt.plot(values, label=name)
            plt.title('Val Metrics History')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()


class UniversalQuantileAggregator(nn.Module):
    def __init__(
        self, 
        n_models: int, 
        quantiles: List[float], 
        weighting_type: str ='global',  # 'global', 'series', 'local'
        resolution: str ='coarse',      # 'coarse', 'medium', 'fine'
        n_series: int =None,            # Нужно, если weighting_type='series'
        x_dim: int =None,               # Нужно, если weighting_type='local'
        hidden_dim: int =64,            # Для MLP (local)
        dropout: float =0.1
    ):
        """
        Универсальный агрегатор, реализующий стратегии из статьи.
        
        Args:
            n_models (int): Число базовых моделей (p).
            n_quantiles (int): Число квантилей (m).
            weighting_type (str): 'global' (статичные веса), 'series' (embedding), 'local' (MLP/DQA).
            resolution (str): 
                'coarse' - скалярный вес на модель.
                'medium' - вектор весов на модель (по квантилям).
                'fine'   - матрица весов (mixing всех квантилей).
            n_series (int): Кол-во рядов (для Embedding).
            x_dim (int): Размерность входных фичей (для MLP).
        """
        super().__init__()
        self.n_models = n_models
        self.weighting_type = weighting_type
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.resolution = resolution
        
        # 1. Определение размерности выхода генератора весов
        # Нам нужно сгенерировать "сырые" логиты, которые потом нарежем
        if resolution == 'coarse':
            # 1 вес на модель
            self.output_dim = n_models 
        elif resolution == 'medium':
            # 1 вес на модель * n_quantiles
            self.output_dim = n_models * self.n_quantiles
        elif resolution == 'fine':
            # 1 вес на модель * n_quantiles (target) * n_quantiles (input)
            self.output_dim = n_models * self.n_quantiles * self.n_quantiles
        else:
            raise ValueError(f"Unknown resolution: {resolution}")

        # 2. Создание генератора весов (Weight Generator)
        if weighting_type == 'global':
            # Просто обучаемые параметры, не зависят от входа
            self.weight_generator = nn.Parameter(torch.zeros(self.output_dim))
            
        elif weighting_type == 'series':
            if n_series is None: raise ValueError("n_series required for 'series' weighting")
            # Lookup table
            self.weight_generator = nn.Embedding(n_series, self.output_dim)
            nn.init.normal_(self.weight_generator.weight, mean=0, std=0.01)

        elif weighting_type == 'local':
            if x_dim is None: raise ValueError("x_dim required for 'local' weighting")
            # MLP (DQA architecture)
            self.weight_generator = nn.Sequential(
                nn.Linear(x_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim)
            )
            self._init_local_weights()
        else:
            raise ValueError(f"Unknown weighting_type: {weighting_type}")

    def _init_local_weights(self):
        """
        Инициализация весов для MLP (Local Aggregation).
        Скрытые слои: Kaiming/Xavier.
        Последний слой: Веса около нуля, чтобы старт был с усреднения моделей.
        """
        for m in self.weight_generator.modules():
            if isinstance(m, nn.Linear):
                # Для скрытых слоев - стандартная хорошая практика (He initialization)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Для ПОСЛЕДНЕГО линейного слоя (который выдает логиты весов)
        # мы хотим, чтобы он выдавал значения близкие к 0.
        last_layer = self.weight_generator[-1]
        nn.init.normal_(last_layer.weight, mean=0, std=0.01)
        if last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, 0)

    def get_logits(self, context_data):
        """Извлекает логиты весов в зависимости от типа агрегации."""
        if self.weighting_type == 'global':
            # context_data игнорируется, но нужно добавить batch dimension
            # context_data предполагается как тензор заглушка или None, 
            # но нам нужен размер батча. Пусть context_data - это base_preds.
            batch_size = context_data.shape[0]
            # [Output_Dim] -> [Batch, Output_Dim]
            return self.weight_generator.unsqueeze(0).expand(batch_size, -1)
            
        elif self.weighting_type == 'series':
            # context_data должнен быть tensor(series_ids) [Batch]
            return self.weight_generator(context_data)
            
        elif self.weighting_type == 'local':
            # context_data должен быть tensor(features) [Batch, X_dim]
            return self.weight_generator(context_data)

    def forward(self, base_preds, context_data=None):
        """
        Args:
            base_preds: [Batch, N_models, N_quantiles]
            context_data: 
                - None (если global, но лучше передать base_preds для определения batch_size)
                - Series IDs [Batch] (если series)
                - Features [Batch, X_dim] (если local)
        """
        # Если Global, используем размер батча из предиктов
        if self.weighting_type == 'global': 
            context_input = base_preds 
        else:
            context_input = context_data

        # 1. Получаем сырые логиты: [Batch, Output_Dim]
        logits = self.get_logits(context_input)
        
        # 2. Обработка в зависимости от разрешения (Resolution)
        
        if self.resolution == 'coarse':
            # Логиты: [Batch, N_models]
            # Softmax по моделям. Сумма весов моделей = 1.
            weights = F.softmax(logits, dim=1) # [Batch, N_models]
            
            # Агрегация: Сумма(Вес_модели * Предикт_модели)
            # Weights: [B, N_models, 1]
            # Preds:   [B, N_models, N_quantiles]
            final_pred = (base_preds * weights.unsqueeze(2)).sum(dim=1)
            
        elif self.resolution == 'medium':
            # Логиты: [Batch, N_models * N_quantiles] -> [Batch, N_models, N_quantiles]
            logits = logits.view(-1, self.n_models, self.n_quantiles)
            
            # Softmax по моделям для каждого квантиля отдельно!
            # То есть для q=0.5 сумма весов моделей должна быть 1.
            weights = F.softmax(logits, dim=1) # [Batch, N_models, N_quantiles]
            
            # Агрегация (Element-wise multiplication)
            final_pred = (base_preds * weights).sum(dim=1)
            
        elif self.resolution == 'fine':
            # Это самая сложная часть (DQA Fine)
            # Логиты: [Batch, N_models * N_out_Q * N_in_Q]
            # Reshape -> [Batch, N_models, N_out_Q, N_in_Q]
            logits = logits.view(-1, self.n_models, self.n_quantiles, self.n_quantiles)
            
            # Softmax: Сумма весов должна быть 1 для каждого выходного квантиля.
            # В статье (Section 3) сказано: сумма по всем моделям И всем входным квантилям = 1.
            # То есть flatten по (Models, In_Q)
            
            # Переставим размерности для удобства softmax: [Batch, Out_Q, Models, In_Q]
            logits_perm = logits.permute(0, 2, 1, 3)
            # Flatten последних двух: [Batch, Out_Q, (Models * In_Q)]
            shape_before = logits_perm.shape
            logits_flat = logits_perm.reshape(shape_before[0], shape_before[1], -1)
            
            weights_flat = F.softmax(logits_flat, dim=2)
            
            # Возвращаем форму: [Batch, Models, Out_Q, In_Q] (после обратной перестановки)
            weights = weights_flat.view(shape_before).permute(0, 2, 1, 3)
            
            # Агрегация: Einsum
            # b: batch
            # m: models
            # o: output quantiles
            # i: input quantiles
            # weights: b m o i
            # preds:   b m i
            # result:  b o (суммируем по m и i)
            final_pred = torch.einsum('bmoi,bmi->bo', weights, base_preds)
            
        return final_pred

    @torch.no_grad()
    def predict_ensemble(self, model_adapters, input_data, context_input, horizon, post_sort=True):
        """
        Универсальный метод для продакшна.
        
        Args:
            context_input: ID ряда (int) или вектор фичей (np.array), или None.
            post_sort (bool): Применять ли сортировку квантилей (рекомендуется статьей).
        """
        self.eval()
        
        # 1. Сбор прогнозов
        base_preds_list = []
        for adapter in model_adapters:
            # [Horizon, N_quantiles]
            preds = adapter(input_data, horizon)
            base_preds_list.append(preds)
            
        # [1, N_models, Horizon, N_quantiles]
        base_preds_np = np.stack(base_preds_list, axis=0)
        base_preds_tensor = torch.tensor(base_preds_np, dtype=torch.float32).unsqueeze(0)
        
        # 2. Подготовка контекста
        # Нам нужно продублировать контекст на весь Horizon, если это Local features
        # Но обычно веса учатся либо на шаг, либо на ряд.
        # В этой реализации считаем, что веса одни на весь горизонт прогноза (статичны во времени)
        
        tensor_context = None
        if self.weighting_type == 'series':
            # [1] -> для Embedding
            tensor_context = torch.tensor([context_input], dtype=torch.long)
        elif self.weighting_type == 'local':
            # [1, X_dim] -> для MLP
            tensor_context = torch.tensor(context_input, dtype=torch.float32).unsqueeze(0)
        # Если global, tensor_context останется None, forward обработает это
            
        # 3. Получаем веса и агрегируем
        # base_preds имеет размер [1, Models, Horizon, Quantiles]
        # А forward ожидает [Batch, Models, Quantiles].
        # Нам нужно временно слить Batch и Horizon, или изменить forward.
        # Проще слить: Batch=1 * Horizon
        
        b, m, h, q = base_preds_tensor.shape
        base_preds_flat = base_preds_tensor.permute(0, 2, 1, 3).reshape(b*h, m, q)
        
        # Контекст тоже нужно расширить на b*h
        if tensor_context is not None:
            # [1, ...] -> [Horizon, ...]
            if self.weighting_type == 'series':
                tensor_context = tensor_context.repeat(h)
            elif self.weighting_type == 'local':
                tensor_context = tensor_context.repeat(h, 1)

        # Вызов forward
        final_forecast_flat = self.forward(base_preds_flat, tensor_context)
        
        # Reshape обратно [Horizon, Quantiles]
        final_forecast = final_forecast_flat.reshape(h, q)
        
        # 4. Post-processing (Sort) - Proposition 2 из статьи
        if post_sort:
            final_forecast, _ = torch.sort(final_forecast, dim=1)
            
        return final_forecast.cpu().numpy()


class TimeSeriesAggregatorPipeline:
    def __init__(self, device='cpu'):
        self.device = device
        self.scaler_y = StandardScaler()

    def __make_model_methods(model) -> Tuple[Callable]:
        """
        This method generates fit, predict_quantiles and update functions for model according to this type.
        E.g. fit funciton for BaseForecaster from sktime is lambda y, X: model.fit(y=y, X=X).
        
        :param model: Instance of model class, e.g. AutoArima, ETS, Prophet.
        
        :return tuple of fit, predict_quantiles and update.
        """
        if isinstance(model, BaseForecaster):
            pass
        

    def generate_ts_oof(self, y, model_factories: dict, X=None, n_splits: int = 5):
        """
        Генерация OOF предсказаний методом TimeSeriesSplit.
        
        Args:
            X, y: Обучающие данные (numpy arrays).
            model_factories: Словарь {name: factory_func}, создающий НОВЫЕ модели.
                             Пример: {'catboost': lambda: CatBoostRegressor(...)}
                             
        Returns:
            
        """
        if not isinstance(model_factories, dict) or not all(map(callable, model_factories.values())):
            raise ValueError(f"model_factories should be a dict of model constructors")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Списки для накопления результатов (так как мы идем по времени)
        oof_quantile_preds_chunks = []
        y_chunks = []

        # Словарь OOF предсказаний моделей для обучения агрегатора и агрегации
        models_oof_data = defaultdict(dict)

        print(f"Запуск OOF генерации ({n_splits} splits)...")
        
        # Прогресс бар по фолдам
        # for model_name, factory in model_factories.items():
        #     model = 
        for train_idx, val_idx in tqdm(tscv.split(y), "FOLD", position=0, total=n_splits):
            
            # 1. Разбиение данных
            # Для временных рядов train_idx всегда идут ДО val_idx
            X_tr, X_val = None, None
            if not X is None:
                X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Сохраняем истинные значения для этого куска валидации
            y_chunks.append(y_val)
            
            # Буфер для предсказаний всех моделей на текущем фолде
            # [N_val_samples, N_models, N_quantiles]
            current_fold_quantile_preds = []
            
            for model_name, factory in tqdm(model_factories.items(), "Model", leave=False, total=len(model_factories)):
                print(model_name)
                current_model_quantile_preds = []

                # 2. Обучение и прогноз базовой модели
                model = factory()
                
                # Обучаем на прошлом
                model.fit(y=y_tr, X=X_tr)
                            
                for start_step in range(0, len(y_val), cfg.FORECAST_HORIZON):
                    end_step = min(start_step + cfg.FORECAST_HORIZON, len(y_val))
                    steps_to_predict = end_step - start_step
                    
                    pred = model.predict_quantiles(fh=np.arange(1, steps_to_predict + 1), X=X_val, alpha=cfg.QUANTILES)
                    current_model_quantile_preds.append(pred)

                    y_new = y_val[start_step : end_step]
                    model.update(y_new, update_params=True)

                
                current_model_quantile_preds = np.concatenate(current_model_quantile_preds, axis=0)
                current_fold_quantile_preds.append(current_model_quantile_preds)
            
            # Склеиваем модели: [N_val, 1, Q] -> [N_val, N_models, Q]
            # stack по оси 1
            fold_quantile_preds_np = np.stack(current_fold_quantile_preds, axis=1)
            oof_quantile_preds_chunks.append(fold_quantile_preds_np)
            
        # 3. Объединение всех чанков в один массив
        # Мы склеиваем по оси времени (axis 0)
        oof_quantile_preds_full = np.concatenate(oof_quantile_preds_chunks, axis=0)
        y_aligned = np.concatenate(y_chunks, axis=0)
        
        print(f"OOF генерация завершена.")
        print(f"Исходный размер: {len(y)}")
        print(f"Размер для агрегации (без первого трейн-куска): {len(y_aligned)}")
        print(f"Размер квантильных прогнозов: {oof_quantile_preds_full.shape}")
        
        for model_idx, model_name in enumerate(model_factories):
            for q_idx, q in enumerate(cfg.QUANTILES):
                models_oof_data[model_name][q] = oof_quantile_preds_full[:, model_idx, q_idx]

        return models_oof_data, oof_quantile_preds_full, y_aligned

    def train_aggregator(self, 
                         oof_preds, 
                         y_true, 
                         aggregator_params: dict,
                         batch_size=256, 
                         epochs=50,
                         lr=1e-3):
        """
        Обучение UniversalQuantileAggregator на OOF прогнозах.
        """
        assert oof_preds.shape[2] == len(cfg.QUANTILES), "oof_preds 2 dimension size should be equal to quantiles in config"
        # 1. Стандартизация таргета (как в статье)
        # Важно фитить скалер только на том, на чем учим агрегатор
        split_idx = int(len(y_true) * 0.8)
        # self.scaler_y.fit(y_true[:split_idx][:, None])
        # y_scaled = self.scaler_y.fit_transform(y_true.reshape(-1, 1)).flatten()
        # y_scaled = self.scaler_y.transform(y_true.reshape(-1, 1)).flatten()
        y_scaled = y_true
        
        # 2. Подготовка тензоров
        # oof_preds: [Samples, Models, Quantiles]
        X_tensor = torch.tensor(oof_preds, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)
        
        # Создаем Dataset
        # Для Global/Coarse нам не нужен context_data, но тренер ждет 2 или 3 аргумента.
        # Передадим X_tensor второй раз как заглушку (тренер разберется)
        
        # Разделяем на Train/Val для самого агрегатора (например, последние 20% времени)
        # split_idx = int(len(y_scaled) * 0.8)
        
        train_ds = TensorDataset(X_tensor[:split_idx], X_tensor[:split_idx], y_tensor[:split_idx])
        val_ds = TensorDataset(X_tensor[split_idx:], X_tensor[split_idx:], y_tensor[split_idx:])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # 3. Инициализация модели
        aggregator = UniversalQuantileAggregator(
            n_models=oof_preds.shape[1],
            quantiles=cfg.QUANTILES,
            **aggregator_params
        )
        
        # 4. Тренер
        optimizer = torch.optim.Adam(aggregator.parameters(), lr=lr)
        criterion = PinballLoss() # Убедитесь, что PinballLoss доступен
        
        trainer = QuantileAggregatorTrainer(
            model=aggregator,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        # 5. Обучение
        history = trainer.fit(train_loader, val_loader, epochs=epochs, early_stopping=15)
        
        return aggregator, history, trainer

if __name__ == '__main__':
    
    pipeline = TimeSeriesAggregatorPipeline()
    dump_quantiles_data = np.random.randn(108, 2, 5)
    dump_y = np.random.randn(108)
    aggregator_params = {
    "resolution": "coarse",
    "weighting_type": "global"
}
    
    aggregator, history, trainer = pipeline.train_aggregator(dump_quantiles_data,
                                                        dump_y,
                                                        aggregator_params=aggregator_params,
                                                        )
    print(torch.softmax(aggregator.weight_generator.data, dim=0))