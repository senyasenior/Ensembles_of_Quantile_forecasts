import numpy as np
import pandas as pd

def calculate_scale(y_train):
    """
    Вычисляет масштаб ряда (Mean Absolute Difference) для нормализации WIS/Pinball Loss.
    Аналог знаменателя в метрике MASE.
    """
    y_tr = np.array(y_train, dtype=np.float64)
    # Игнорируем NaN, если есть
    y_tr = y_tr[~np.isnan(y_tr)]
    
    if len(y_tr) < 2:
        return 1.0
        
    diff = np.abs(np.diff(y_tr))
    scale = np.mean(diff)
    return scale if scale > 1e-9 else 1.0

def pinball_loss(y_true, y_pred, alpha):
    """
    Рассчитывает Pinball Loss для одного квантиля.
    """
    delta = y_true - y_pred
    sign = (delta >= 0).astype(float)
    loss = (alpha * sign * delta) - ((1 - alpha) * (1 - sign) * delta)
    return np.mean(loss)

def prediction_interval_coverage(y_true, y_lower, y_upper):
    """
    Рассчитывает процент попадания истинных значений в интервал (PICP).
    """
    # y_lower <= y_true <= y_upper
    # Используем <= для включения границ
    hits = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(hits)

def prediction_interval_width(y_lower, y_upper):
    """
    Рассчитывает среднюю ширину интервала (MPIW).
    """
    return np.mean(y_upper - y_lower)

def evaluate_metrics(y_true, y_preds, quantiles, scale=1.0):
    """
    Комплексная оценка прогнозов.
    
    Args:
        y_true (np.array): Истинные значения [N,].
        y_preds (np.array): Предсказанные квантили [N, n_quantiles].
        quantiles (list): Список уровней квантилей.
        scale (float): Коэффициент масштаба ряда.
        
    Returns:
        dict: Словарь с метриками (WIS, Coverage, Calibration Error и т.д.)
    """
    y_true = np.array(y_true)
    y_preds = np.array(y_preds)
    quantiles = np.array(quantiles)
    
    metrics = {}
    
    # 1. Pinball Loss & WIS
    # WIS ≈ 2 * Mean(Pinball Loss) over all quantiles
    total_pinball = 0.0
    pinball_per_q = {}
    
    # Считаем калибровку (Coverage)
    # Coverage считается кумулятивно: P(y <= q_pred) ≈ q
    observed_coverage = []
    
    for i, q in enumerate(quantiles):
        q_pred = y_preds[:, i]
        loss = pinball_loss(y_true, q_pred, q)
        
        # Нормализация
        scaled_loss = loss / scale
        
        pinball_per_q[q] = scaled_loss
        total_pinball += scaled_loss
        
        # Calibration (ACE - Average Calibration Error)
        cov = np.mean(y_true <= q_pred)
        observed_coverage.append(cov)

    avg_pinball_scaled = total_pinball / len(quantiles)
    metrics['WIS'] = 2 * avg_pinball_scaled # По формуле из статьи
    metrics['Mean_Pinball'] = avg_pinball_scaled
    
    # 2. Calibration Metrics
    # MACE: Mean Absolute Calibration Error (насколько реальная частота отличается от теоретической alpha)
    mace = np.mean(np.abs(np.array(observed_coverage) - quantiles))
    metrics['MACE'] = mace
    
    # 3. Interval Metrics (Coverage & Width)
    # Работает только для симметричных пар квантилей вокруг 0.5 (например, 0.1 и 0.9)
    # Находим пары
    intervals = {}
    sorted_indices = np.argsort(quantiles)
    
    # Ищем пары (low, high)
    n_q = len(quantiles)
    mid = n_q // 2
    
    # Если есть медиана (нечетное число квантилей), она не образует интервал
    
    for i in range(mid):
        # 0.1 (idx 0) и 0.9 (idx -1)
        low_idx = sorted_indices[i]
        high_idx = sorted_indices[-(i+1)]
        
        q_low = quantiles[low_idx]
        q_high = quantiles[high_idx]
        
        # Номинальное покрытие (например 0.9 - 0.1 = 0.8)
        nominal_coverage = q_high - q_low
        
        actual_coverage = prediction_interval_coverage(
            y_true, y_preds[:, low_idx], y_preds[:, high_idx]
        )
        
        width = prediction_interval_width(y_preds[:, low_idx], y_preds[:, high_idx])
        scaled_width = width / scale
        
        intervals[f'Int_{nominal_coverage:.2f}'] = {
            'Coverage': actual_coverage,
            'Cov_Error': actual_coverage - nominal_coverage,
            'Width_Scaled': scaled_width
        }
    
    metrics['Intervals'] = intervals
    
    return metrics