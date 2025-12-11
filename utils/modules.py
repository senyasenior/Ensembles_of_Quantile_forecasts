import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Callable, Optional, List, Tuple
import matplotlib.pyplot as plt
import os

import time
import pandas as pd
from catboost import CatBoostRegressor
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from utils.config import cfg
from collections import defaultdict
from sktime.forecasting.base import BaseForecaster
from statsmodels.tsa.stattools import adfuller
import warnings

class SktimeProductionAdapter:
    def __init__(self, model, quantiles):
        """
        Адаптер для использования моделей sktime внутри UniversalQuantileAggregator.
        
        Args:
            model: Обученная модель sktime (должна иметь метод predict_quantiles).
            quantiles: Список квантилей, например [0.1, 0.5, 0.9].
                       Важно: порядок квантилей в выходном массиве будет соответствовать
                       порядку, в котором sktime их возвращает (обычно отсортированный).
                       Убедитесь, что Aggregator обучен на таком же порядке.
        """
        self.model = model
        self.quantiles = quantiles

    def __call__(self, input_data, horizon):
        """
        Вызывается внутри predict_ensemble.
        
        Args:
            input_data: Данные для прогноза. 
                        Если модели нужны экзогенные переменные (X), input_data может быть:
                        1. Словарем {'X': pd.DataFrame/np.array}
                        2. Самим DataFrame/np.array (X)
                        Если модель одномерная и не требует X, можно передавать None.
            horizon (int): Горизонт прогнозирования (целое число).
            
        Returns:
            np.array: Массив размера [Horizon, N_quantiles]
        """
        
        fh = np.arange(1, horizon + 1)

        # Обработка входных данных (Exogenous variables)
        X = None
        if input_data is not None:
            if isinstance(input_data, dict) and 'X' in input_data:
                # Если передан словарь с ключом 'X'
                X = input_data['X']
            else:
                # Если передано что-то другое, считаем это X
                X = input_data

        
        pred_df = self.model.predict_quantiles(fh=fh, X=X, alpha=self.quantiles)
        return pred_df.values


class CatBoostRecursiveWrapper:
    def __init__(self, 
                 quantiles, 
                 lags, 
                 rolling_windows, 
                 diff_periods, 
                 val_size=None,
                 log_transform=False,
                 auto_diff=False,        
                 max_diff_order=2,       
                 adf_p_value=0.05,       
                 **catboost_params):
        
        self.quantiles = sorted(quantiles)
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.diff_periods = diff_periods
        self.val_size = val_size
        
        # Настройки предобработки
        self.log_transform = log_transform
        self.auto_diff = auto_diff
        self.max_diff_order = max_diff_order
        self.adf_p_value = adf_p_value
        
        # Внутреннее состояние
        self.diff_order_ = 0       
        self.last_values_ = []     
        self.history = None        # История (уже трансформированная и стационарная!)
        self.model = None
        
        # CatBoost init
        self.catboost_params = catboost_params
        q_str = ",".join([str(q) for q in self.quantiles])
        self.loss_function = f'MultiQuantile:alpha={q_str}'
        
        if 0.5 in self.quantiles:
            self.median_idx = self.quantiles.index(0.5)
        else:
            self.median_idx = len(self.quantiles) // 2

    def _check_stationarity(self, series):
        """Проверяет ряд на стационарность через Augmented Dickey-Fuller test."""
        # ADF требует определенной длины, если ряд короткий - пропускаем
        if len(series) < 10:
            return True 
            
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = adfuller(series)
            return result[1] < self.adf_p_value
        except:
            return False # Если тест упал, считаем нестационарным

    def _transform(self, y, mode='fit'):
        """
        Применяет Log -> Diff (iterative).
        mode='fit': определяет порядок d и сохраняет якоря.
        mode='update': использует сохраненный d и обновляет якоря.
        """
        y_trans = np.array(y, dtype=float)
        
        # 1. Log Transform
        if self.log_transform:
            y_trans = np.log1p(y_trans)
            
        # 2. Auto Differencing
        if self.auto_diff:
            if mode == 'fit':
                self.diff_order_ = 0
                self.last_values_ = [] # Стек последних значений для каждого уровня diff
                
                # Итеративно дифференцируем
                current_series = y_trans.copy()
                
                for d in range(self.max_diff_order):
                    # Проверка
                    if self._check_stationarity(current_series):
                        break
                    
                    # Сохраняем последнее значение ПЕРЕД дифференцированием (якорь для восстановления)
                    self.last_values_.append(current_series[-1])
                    
                    # Применяем diff
                    current_series = np.diff(current_series)
                    self.diff_order_ += 1
                    
                y_trans = current_series
                
            elif mode == 'update':
                # При update мы просто применяем УЖЕ ВЫУЧЕННЫЙ порядок diff
                # И обновляем якоря, чтобы прогноз шел от конца новых данных
                
                # Нам нужно сохранить последние значения для восстановления БУДУЩЕГО прогноза.
                # Поэтому мы должны "прогнать" новые данные через процесс и обновить self.last_values_
                
                # Сложность: last_values_ должны быть "последними известными абсолютными значениями"
                # на конец всего ряда (старая история + новая).
                
                # Восстанавливаем полную картину (или берем хвост, если он длинный)
                # Проще всего: при update мы обновляем last_values_ глядя на "сырой" y (после логарифма)
                
                temp_series = y_trans.copy() # Это уже логарифмированные данные
                
                # Обновляем якоря
                # Якорь уровня 0 = последнее значение ряда
                # Якорь уровня 1 = последнее значение diff(ряда)
                # ...
                new_anchors = []
                for _ in range(self.diff_order_):
                    new_anchors.append(temp_series[-1])
                    temp_series = np.diff(temp_series)
                
                self.last_values_ = new_anchors
                
                # Результат трансформации для дообучения
                y_trans = np.diff(y_trans, n=self.diff_order_)

        return y_trans

    def _inverse_transform(self, preds_matrix, horizon):
        """
        Восстанавливает прогноз: Cumsum (d раз) -> Expm1.
        preds_matrix: [Horizon, Quantiles] - это предсказанные ИЗМЕНЕНИЯ.
        """
        preds_restored = preds_matrix.copy()
        
        # 1. Inverse Diff (идем с конца: от d-го порядка к 0-му)
        if self.auto_diff and self.diff_order_ > 0:
            # last_values_ хранит [val_d0, val_d1, ...]
            # Нам нужно применять их в обратном порядке: сначала восстановить d-1 из d, потом d-2...
            
            # Пример: d=1. preds - это diffs.
            # restored = cumsum(preds) + last_val
            
            for d in reversed(range(self.diff_order_)):
                anchor = self.last_values_[d]
                # Cumsum по оси времени (axis=0)
                preds_restored = np.cumsum(preds_restored, axis=0) + anchor
        
        # 2. Inverse Log
        if self.log_transform:
            preds_restored = np.expm1(preds_restored)
            
        return preds_restored

    def _generate_features(self, y_data):
        # ... (стандартная генерация фичей без изменений) ...
        # Копируем из предыдущего ответа
        if not isinstance(y_data, pd.Series):
            if isinstance(y_data, np.ndarray):
                df = pd.DataFrame({'target': y_data})
            else:
                df = pd.DataFrame({'target': y_data})
        else:
            df = pd.DataFrame({'target': y_data.values}, index=y_data.index)
            
        # Тренд (важен!)
        df['trend_idx'] = np.arange(len(df))
        
        for lag in self.lags:
            df[f'lag_{lag}'] = df['target'].shift(lag)

        base_col = df['target'].shift(1)
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = base_col.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = base_col.rolling(window=window).std()

        for period in self.diff_periods:
            df[f'diff_{period}'] = df['target'].diff(period)

        df = df.dropna()
        y = df['target']
        X = df.drop(columns=['target'])
        return X, y

    def fit(self, y, X=None):
        # 1. Трансформация (Log + AutoDiff)
        # fit определяет self.diff_order_ и заполняет self.last_values_
        y_trans = self._transform(y, mode='fit')
        
        # Сохраняем ИСТОРИЮ в стационарном виде для генерации фичей
        self.history = list(y_trans)
        
        # 2. Генерация фичей на стационарном ряде
        X_full, y_full = self._generate_features(pd.Series(y_trans))
        
        # 3. Параметры
        params = {
            'iterations': 1000,
            'verbose': 0,
            'allow_writing_files': False,
            'random_state': 42,
            'task_type': "CPU"
        }
        params.update(self.catboost_params)
        params['loss_function'] = self.loss_function
        
        # 4. Train/Eval Split
        eval_set = None
        if self.val_size and self.val_size > 0:
            if len(X_full) <= self.val_size:
                X_train, y_train = X_full, y_full
            else:
                X_train = X_full.iloc[:-self.val_size]
                y_train = y_full.iloc[:-self.val_size]
                X_eval = X_full.iloc[-self.val_size:]
                y_eval = y_full.iloc[-self.val_size:]
                eval_set = (X_eval, y_eval)
        else:
            X_train, y_train = X_full, y_full

        # 5. Обучение
        self.model = CatBoostRegressor(**params)
        self.model.fit(X_train, y_train, eval_set=eval_set)
        
        # Вывод информации о трансформации (для отладки)
        # print(f"Diff Order: {self.diff_order_}, Log: {self.log_transform}")
        
        return self

    def predict_quantiles(self, fh, X=None, alpha=None) -> pd.DataFrame:
        horizon = fh if isinstance(fh, int) else len(fh)
        
        # Рекурсия идет на СТАЦИОНАРНЫХ данных
        current_history = self.history.copy()
        future_preds_stationary = []
        
        max_lookback = 0
        if self.lags: max_lookback = max(max_lookback, max(self.lags))
        if self.rolling_windows: max_lookback = max(max_lookback, max(self.rolling_windows))
        if self.diff_periods: max_lookback = max(max_lookback, max(self.diff_periods))
        max_lookback += 5 
        
        for _ in range(horizon):
            recent_history = current_history[-max_lookback:]
            temp_hist = recent_history + [0] 
            
            # Генерация фичей (на стационарном ряде)
            X_temp, _ = self._generate_features(pd.Series(temp_hist))
            
            if X_temp.empty: raise ValueError("Not enough history")
            X_next = X_temp.iloc[[-1]]
            
            # Коррекция trend_idx для стационарного ряда (он продолжает расти)
            X_next = X_next.copy()
            X_next['trend_idx'] = len(current_history)
            
            # Предсказание (это изменения/diffs, если d>0)
            pred_q = self.model.predict(X_next)
            future_preds_stationary.append(pred_q[0])
            
            # Рекурсия через медиану
            pred_median = pred_q[0, self.median_idx]
            current_history.append(pred_median)
            
        # --- ВОССТАНОВЛЕНИЕ (Inverse Transform) ---
        preds_matrix = np.array(future_preds_stationary)
        
        # Здесь мы используем self.last_values_, которые были сохранены в fit() или update()
        # Они содержат абсолютные значения на КОНЕЦ обучающей выборки.
        # Это позволяет корректно приклеить cumsum к концу трейна.
        final_preds = self._inverse_transform(preds_matrix, horizon)
        
        # Индексы
        if hasattr(fh, '__iter__'):
            index = fh
        else:
            index = np.arange(1, horizon + 1)
            
        return pd.DataFrame(final_preds, index=index, columns=self.quantiles)

    def update(self, y_new, X_new=None, update_params=True):
        # 1. Сначала обновляем self.last_values_ и получаем стационарный кусок
        # Важно: нам нужно передать 'update' в transform, чтобы он использовал старый d
        # и обновил якоря. Но _transform ожидает на вход ВЕСЬ ряд или кусок?
        # В нашей логике update_anchors требует контекста.
        
        # Проще всего: Склеить старую "сырую" историю (которой у нас нет, мы храним stationary)
        # А, мы не храним сырую историю. Это проблема для update якорей.
        
        # РЕШЕНИЕ: При update мы должны подавать "сырые" y_new.
        # Но чтобы обновить якорь (последнее значение тренда), нам нужно последнее значение
        # предыдущего куска.
        # self.last_values_ уже содержит последнее значение предыдущего куска (anchor).
        
        # Давайте обновим last_values_ "на лету", проходя по y_new
        y_values = np.array(y_new, dtype=float)
        if self.log_transform:
            y_values = np.log1p(y_values)
            
        # Обновление якорей
        # Если d=1, то новый якорь = последнее значение y_new
        # Если d=2, то новый якорь[0] = последнее y_new
        #           новый якорь[1] = последнее diff(y_new)
        
        # Но сначала нам нужно превратить y_new в стационарный вид для обучения модели
        # Для этого нужно знать "стык" между старой историей и y_new.
        # Стыки лежат в self.last_values_.
        
        y_trans = y_values.copy()
        
        # Итеративное дифференцирование с учетом стыка
        # Для d=0..max:
        #   y_diff = y[t] - y[t-1].
        #   Для первого элемента y_new[0] нам нужно prev_last_value.
        
        new_anchors = []
        
        for d in range(self.diff_order_):
            prev_anchor = self.last_values_[d]
            
            # Сохраняем новый якорь для этого уровня (последнее значение ТЕКУЩЕГО ряда)
            new_anchors.append(y_trans[-1])
            
            # Дифференцируем y_trans
            # np.diff теряет 1 элемент. Но у нас есть prev_anchor!
            # diff[0] = y_trans[0] - prev_anchor
            # diff[1..] = np.diff(y_trans)
            
            first_diff = y_trans[0] - prev_anchor
            rest_diff = np.diff(y_trans)
            y_trans = np.concatenate([[first_diff], rest_diff])
            
        # Обновляем якоря класса на новые (теперь они указывают на конец y_new)
        if self.diff_order_ > 0:
            self.last_values_ = new_anchors
            
        # Добавляем стационарный кусок в историю
        self.history.extend(list(y_trans))
        
        # Дообучение
        if update_params:
            # Генерируем фичи на (теперь увеличенной) стационарной истории
            # Берем окно для скорости
            X_tr, y_tr = self._generate_features(pd.Series(self.history[-2000:])) # Пример окна
            self.model.fit(X_tr, y_tr, init_model=self.model)
            
        return self


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

def calculate_adaptive_margins(y_true, models_oof: np.ndarray, quantiles, scale_factor=0.1):
    """
    Реализация формулы для расчета адаптивных margin'ов.
    
    Args:
        y_true: Истинные значения OOF [N_samples]
        models_oof: Словарь или массив прогнозов. 
                    Нужно достать прогнозы для МЕДИАНЫ (0.5).
        quantiles: Список всех квантилей.
    """
    # 1. Pilot Estimator (g_hat_0)
    
    if 0.5 not in quantiles:
        raise ValueError("Для метода из в списке квантилей должна быть медиана (0.5)")
    
    med_idx = quantiles.index(0.5)
    
    median_preds = models_oof[:, :, med_idx] 
    g0 = np.mean(median_preds, axis=1)
        
    # 2. Residuals (R_i)
    residuals = y_true - g0
    
    # 3. Расчет margins 
    margins = []
    

    residual_quantiles_values = np.quantile(residuals, quantiles)
    res_q_map = dict(zip(quantiles, residual_quantiles_values))
    
    # Идем по парам соседних квантилей (tau, tau')
    for i in range(len(quantiles) - 1):
        tau = quantiles[i]
        tau_prime = quantiles[i+1]
        
        # Gap = Q_tau'(R) - Q_tau(R)
        gap = res_q_map[tau_prime] - res_q_map[tau]
        
        #delta = delta_0 * (gap)+
        margin_val = scale_factor * max(0, gap)
        margins.append(margin_val)
        
    return torch.tensor(margins, dtype=torch.float32)

class PenalizedPinballLoss:
    def __init__(
        self, 
        quantiles=None,       
        penalty_weight=0.0,   
        margins=None          
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        q_list = quantiles if quantiles is not None else cfg.QUANTILES
        self.quantiles = torch.tensor(q_list, dtype=torch.float32, device=device)
        
        self.penalty_weight = penalty_weight
        
        if margins is not None:
            if not isinstance(margins, torch.Tensor):
                self.margins = torch.tensor(margins, dtype=torch.float32, device=device)
            else:
                self.margins = margins.to(device)
        else:
            self.margins = None

    def __call__(self, predict_data: torch.Tensor, target_data: torch.Tensor):
        """
        predict_data: [Batch, N_quantiles]
        target_data: [Batch, 1] (или [Batch])
        """
        if self.quantiles.device != predict_data.device:
            self.quantiles = self.quantiles.to(predict_data.device)
        
        # --- 1. Основной Pinball Loss (Ваш код) ---
        if target_data.dim() == 1:
            target_data = target_data.view(-1, 1)
            
        error_data = target_data - predict_data 
        loss_data = torch.max(self.quantiles * error_data, (self.quantiles - 1) * error_data)
        total_pinball = loss_data.sum()
        
        # --- 2. Crossing Penalty (Штраф за пересечение/сближение) ---
        if self.penalty_weight > 0:

            diffs = predict_data[:, 1:] - predict_data[:, :-1]
            
            if self.margins is not None:
                if self.margins.device != predict_data.device:
                    self.margins = self.margins.to(predict_data.device)
                penalty_terms = torch.relu(self.margins - diffs)
            else:
                penalty_terms = torch.relu(-diffs)
            
            crossing_loss = penalty_terms.sum()
            
            return total_pinball + self.penalty_weight * crossing_loss
            
        return total_pinball


class QuantileAggregatorTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable, # PinballLoss | PenalizedPinballLoss
        device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
        compile_model: bool = False, 
        enable_amp: bool = False,     
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

        #callable объекты для предсказания квантилей в продакшне
        

        print(f"Запуск OOF генерации ({n_splits} splits)...")
        
        # Прогресс бар по фолдам
        # for model_name, factory in model_factories.items():
        #     model = 
        for i_fold, (train_idx, val_idx) in enumerate(tqdm(tscv.split(y), "FOLD", position=0, total=n_splits)):
            
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
                current_model_quantile_preds = []

                # 2. Обучение и прогноз базовой модели
                model = factory()
                
                # Обучаем на прошлом
                
                if X_tr is not None:
                    model.fit(y=y_tr, X=X_tr)
                else: model.fit(y=y_tr)
                            
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
        print(f"Размер квантильных прогнозов: {oof_quantile_preds_full.shape}\n")
        
        for model_idx, model_name in enumerate(model_factories):
            for q_idx, q in enumerate(cfg.QUANTILES):
                models_oof_data[model_name][q] = oof_quantile_preds_full[:, model_idx, q_idx].tolist()

        return models_oof_data, oof_quantile_preds_full, y_aligned

    def train_aggregator(self, 
                         oof_preds, 
                         y_true, 
                         aggregator_params: dict,
                         criterion: PinballLoss|PenalizedPinballLoss, 
                         batch_size=256, 
                         epochs=50,
                         lr=1e-3):
        """
        Обучение UniversalQuantileAggregator на OOF прогнозах.
        """
        assert oof_preds.shape[2] == len(cfg.QUANTILES), "oof_preds 2 dimension size should be equal to quantiles in config"
        assert isinstance(criterion, PenalizedPinballLoss) or isinstance(criterion, PinballLoss),\
        f"critreion must be an istance of PinballLoss or PenalizedPinballLoss. {type(criterion)} given"
        
        split_idx = int(len(y_true) * 0.8)
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
        
        trainer = QuantileAggregatorTrainer(
            model=aggregator,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        # 5. Обучение
        history = trainer.fit(train_loader, val_loader, epochs=epochs, early_stopping=100)
        
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