import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from itertools import combinations
from sklearn.metrics import mean_pinball_loss
from utils.metrics import calculate_scale, evaluate_metrics
from utils.modules import UniversalQuantileAggregator, QuantileAggregatorTrainer, PinballLoss 




def evaluate_model_combinations(y_true, models_dict, quantiles, y_train=None, return_raw=False):
    """
    Сравнивает одиночные модели и их ансамбли (винцентизация) по метрике Pinball Loss.

    Args:
        y_true (array-like): Истинные значения ряда (тестовая выборка).
        models_dict (dict): Словарь вида {
            'ModelName': {0.05: [pred_q05], 0.5: [pred_q50], ...},
            ...
        }
            Здесь ключи - имена моделей, значения - словари, где ключ - квантиль, 
            значение - массив предсказаний.
        quantiles (list): Список проверяемых квантилей (напр. [0.05, 0.5, 0.95]).
        y_train: (Опционально) Обучающая выборка. Если передана,
                 метрика будет Scaled Pinball Loss (как в M5).
                 Если None, то обычный Pinball Loss.
        return_raw: Если True, возвращает сырой DataFrame без стилей (для дальнейшей обработки).
    Returns:
        pd.DataFrame: Таблица с метриками (Stylized DataFrame).
    """
    
    # 1. Расчет скейлера (если передан трейн)
    scale = 1.0
    
    if y_train is not None:
        scale = calculate_scale(y_train)
        print(f"Используется Scale: {scale:.4f}")

    model_names = list(models_dict.keys())
    results = {q: {} for q in quantiles}
    
    # Списки для поиска лучших
    best_single_summary = {}
    best_combo_summary = {}

    # 1. Генерация всех комбинаций (от 1 модели до всех вместе)
    all_combos = []
    for r in range(1, len(model_names) + 1):
        all_combos.extend(combinations(model_names, r))

    # 2. Расчет метрик
    for q in quantiles:
        # Для поиска минимумов внутри конкретного квантиля
        single_scores = {}
        combo_scores = {}
        
        for combo in all_combos:
            # Формируем красивое имя (например, "Arima + Prophet")
            combo_name = " + ".join(combo)
            
            # Собираем предсказания всех моделей в комбинации для текущего квантиля
            preds_list = []
            for model_name in combo:
                # Важно: приводим к numpy array для корректного усреднения
                preds = np.array(models_dict[model_name][q])
                # Проверка размерности, на всякий случай
                assert len(preds) == len(y_true), f"Lenghts of predictions must be equal to y_true. Given predicitons: {len(preds)}, y_true: {len(y_true)}"

                preds_list.append(preds)
            
            # Винцентизация (среднее арифметическое предсказаний квантилей)
            ensemble_pred = np.mean(preds_list, axis=0)
            
            # Считаем ошибку
            loss = mean_pinball_loss(y_true, ensemble_pred, alpha=q)
            loss /= scale
            results[q][combo_name] = loss
            
            # Сохраняем для статистики
            if len(combo) == 1:
                single_scores[combo_name] = loss
            else:
                combo_scores[combo_name] = loss

        # Определяем лучших для текущего квантиля
        best_single = min(single_scores, key=single_scores.get)
        best_single_val = single_scores[best_single]
        
        if combo_scores:
            best_any_combo = min(combo_scores, key=combo_scores.get)
            best_any_val = combo_scores[best_any_combo]
            
            # Сравниваем лучшую одиночную с лучшей комбинацией
            overall_best = best_any_combo if best_any_val < best_single_val else best_single
            overall_val = min(best_any_val, best_single_val)
        else:
            best_any_combo = "Нет комбинаций"
            best_any_val = float('inf')
            overall_best = best_single
            overall_val = best_single_val

        best_single_summary[q] = (best_single, best_single_val)
        best_combo_summary[q] = (overall_best, overall_val)

    # 3. Формируем DataFrame
    # Строки - Квантили, Столбцы - Модели/Комбинации
    df = pd.DataFrame(results)
    
    # Транспонируем, чтобы Квантили были строками (как в ТЗ)
    df = df.T 
    
    # Сортировка столбцов: Сначала одиночные модели, потом комбинации (по длине названия)
    cols = sorted(df.columns, key=lambda x: (len(x.split(' + ')), x))
    df = df[cols]

    # 4. Вывод текстового резюме
    print("="*80)
    print(f"{'Квантиль':<10} | {'Лучшая одиночная модель':<30} | {'Лучшая комбинация (Overall)':<30}")
    print("-" * 80)
    for q in quantiles:
        s_name, s_score = best_single_summary[q]
        c_name, c_score = best_combo_summary[q]
        print(f"{q:<10} | {s_name:<22} ({s_score:.4f}) | {c_name:<22} ({c_score:.4f})")
    print("="*80 + "\n")

    # 5. Стилизация таблицы (подсветка минимума в строке)
    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: #4ef048; font-weight: bold' if v else '' for v in is_min]

    return df.style.apply(highlight_min, axis=1).format("{:.4f}")

def prepare_tensors_from_dict(models_dict, combo_names, quantiles, y_true, series_ids=None):
    """
    Преобразует словарь предсказаний в тензоры PyTorch для агрегатора.
    Возвращает: X [N, Models, Quantiles], y [N, 1], context [N]
    """
    n_samples = len(y_true)
    n_models = len(combo_names)
    n_quantiles = len(quantiles)
    
    # [Samples, Models, Quantiles]
    X_data = np.zeros((n_samples, n_models, n_quantiles))
    
    for m_idx, m_name in enumerate(combo_names):
        if m_name not in models_dict:
            raise ValueError(f"Model '{m_name}' not found in predictions dictionary.")
            
        for q_idx, q in enumerate(quantiles):
            # Предполагаем, что данные лежат как массивы (N,)
            X_data[:, m_idx, q_idx] = models_dict[m_name][q]
            
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.float32).view(-1, 1)
    
    # Context (Series IDs)
    if series_ids is not None:
        # Если передали один ID (int), размножаем его
        if isinstance(series_ids, (int, float)):
             context_tensor = torch.full((n_samples,), series_ids, dtype=torch.long)
        # Если передали массив
        else:
             context_tensor = torch.tensor(series_ids, dtype=torch.long)
    else:
        # Заглушка (Global mode), нули
        context_tensor = torch.zeros(n_samples, dtype=torch.long)
        
    return X_tensor, y_tensor, context_tensor

def train_and_predict_aggregator(
    X_train, y_train, c_train,  # OOF данные (Train)
    X_test, c_test,             # TEST данные (Predict)
    agg_config, 
    n_models, 
    quantiles, 
    device='cpu',
    epochs=30
):
    """
    Обучает агрегатор на OOF и делает прогноз на TEST, 
    полностью повторяя логику predict_ensemble (forward + sort).
    """
    # 1. Разделение OOF на Train/Validation для Early Stopping (80/20)
    split_idx = int(len(y_train) * 0.8)
    
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    c_tr, c_val = c_train[:split_idx], c_train[split_idx:]
    
    train_ds = TensorDataset(X_tr, c_tr, y_tr)
    val_ds = TensorDataset(X_val, c_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    
    # 2. Инициализация модели
    # Для series-specific нам нужно знать макс ID. 
    # Берем макс из обоих сетов (на случай если в тесте новый ID, хотя это странно)
    n_series_max = max(int(c_train.max().item()), int(c_test.max().item())) + 1
    
    model = UniversalQuantileAggregator(
        n_models=n_models,
        quantiles=quantiles,
        weighting_type=agg_config['weighting_type'],
        resolution=agg_config['resolution'],
        n_series=n_series_max,
        dropout=0.1
    )
    
    # 3. Обучение
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = PinballLoss()
    
    trainer = QuantileAggregatorTrainer(
        model, optimizer, criterion, device, 
        checkpoint_dir='checkpoints' # Временная папка
    )
    
    # verbose=False, чтобы не засорять вывод в цикле комбинаций
    trainer.fit(train_loader, val_loader, epochs=epochs, early_stopping=100)
    
    # 4. Предсказание (Логика predict_ensemble)
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        c_test = c_test.to(device)
        
        # А. Forward pass
        # model.forward ожидает [Batch, Models, Quantiles] и [Batch] (Context)
        # У нас X_test уже в таком формате, нам не нужно делать reshape как в predict_ensemble,
        # так как мы подаем сразу весь батч
        preds_tensor = model(X_test, c_test)
        
        # Б. Post-processing (Sort) - КАК В СТАТЬЕ
        # Это ключевой момент из predict_ensemble
        preds_tensor, _ = torch.sort(preds_tensor, dim=1)
        
        final_preds = preds_tensor.cpu().numpy()
        
    return final_preds

def evaluate_model_combinations_advanced(
    y_test,                 
    models_test_dict,       
    y_oof,                  
    models_oof_dict,        
    quantiles, 
    y_train_hist=None,      
    aggregator_configs=[
        {'weighting_type': 'global', 'resolution': 'coarse'},
        {'weighting_type': 'global', 'resolution': 'medium'}
    ],
    epochs: int = 10000,
    series_ids_oof=None,    
    series_ids_test=None,   
    device='cpu',
    metric_mode='WIS' # <--- НОВЫЙ АРГУМЕНТ: 'WIS' или 'Pinball'
):
    """
    Сравнивает модели, их комбинации и агрегаторы.
    
    Args:
        metric_mode (str): 
            'WIS' - выводит таблицу с Weighted Interval Score (одно число на комбинацию).
            'Pinball' - выводит таблицу с Pinball Loss для каждого квантиля отдельно.
    """
    if len(models_test_dict) <= 1: 
        raise Exception("Warning: Only 1 model provided. Aggregation requires at least 2.")
        
    # 1. Расчет скейла
    scale = 1.0
    if y_train_hist is not None:
        scale = calculate_scale(y_train_hist)
        print(f"Scale computed on history: {scale:.4f}")
        
    model_names = list(models_test_dict.keys())
    
    # Мы будем собирать данные в "длинном" формате для гибкости
    # Структура: {'Model': ..., 'Method': ..., 'Metric': ..., 'Value': ...}
    long_results = []
    
    # Вспомогательная функция для расчета и сохранения метрик
    def process_metrics(y_true, y_pred, name, method):
        # 1. Считаем общий WIS
        metrics = evaluate_metrics(y_true, y_pred, quantiles, scale)
        long_results.append({
            'Model': name, 'Method': method, 'Metric': 'WIS', 'Value': metrics['WIS']
        })
        
        # 2. Считаем Pinball Loss для каждого квантиля отдельно
        # L(q) = mean(max(q*e, (q-1)*e)) / scale
        residuals = y_true.reshape(-1, 1) - y_pred
        for i, q in enumerate(quantiles):
            loss = np.maximum(q * residuals[:, i], (q - 1) * residuals[:, i])
            mean_loss_scaled = np.mean(loss) / scale
            
            long_results.append({
                'Model': name, 
                'Method': method, 
                'Metric': f'q_{q}', # Метка для столбца
                'Value': mean_loss_scaled
            })

    # --- ЦИКЛ ПО КОМБИНАЦИЯМ ---
    for r in range(1, len(model_names) + 1):
        for combo in combinations(model_names, r):
            combo_name = " + ".join(combo)
            
            # === A. Vincentization ===
            preds_np_list = []
            for q in quantiles:
                q_preds = []
                for m in combo:
                    q_preds.append(models_test_dict[m][q])
                avg_pred = np.mean(q_preds, axis=0)
                preds_np_list.append(avg_pred)
            
            vinc_preds = np.stack(preds_np_list, axis=1)
            process_metrics(y_test, vinc_preds, combo_name, 'Vincentization')
            
            # === B. Neural Aggregators ===
            if len(combo) > 1:
                X_train_t, y_train_t, c_train_t = prepare_tensors_from_dict(
                    models_oof_dict, combo, quantiles, y_oof, series_ids_oof
                )
                X_test_t, _, c_test_t = prepare_tensors_from_dict(
                    models_test_dict, combo, quantiles, y_test, series_ids_test
                )
                
                for conf in aggregator_configs:
                    agg_name = f"Agg_{conf['weighting_type']}-{conf['resolution']}"
                    try:
                        agg_preds = train_and_predict_aggregator(
                            X_train_t, y_train_t, c_train_t,
                            X_test_t, c_test_t,
                            agg_config=conf,
                            n_models=len(combo),
                            quantiles=quantiles,
                            device=device,
                            epochs=epochs
                        )
                        process_metrics(y_test, agg_preds, combo_name, agg_name)
                        
                    except Exception as e:
                        print(f"Error training {agg_name} for {combo_name}: {e}")

    # --- ФОРМИРОВАНИЕ ОТЧЕТА ---
    if not long_results:
        return pd.DataFrame()

    df_long = pd.DataFrame(long_results)
    
    # Логика отображения в зависимости от режима
    if metric_mode == 'WIS':
        # Фильтруем только WIS
        df_wis = df_long[df_long['Metric'] == 'WIS']
        pivot_table = df_wis.pivot(index='Model', columns='Method', values='Value')
        # Сортируем по качеству Vincentization
        # pivot_table = pivot_table.sort_values(by='Vincentization')
        print("\n=== FINAL RESULTS (Weighted Interval Score) ===")
        
    elif metric_mode == 'Pinball':
        # Фильтруем все квантили (исключаем WIS)
        df_pin = df_long[df_long['Metric'] != 'WIS']
        
        # Строим MultiIndex колонки: Method -> Metric (Quantile)
        # Это создаст таблицу: 
        #           | Vincentization      | Agg_global-coarse   | ...
        # Model     | q_0.1 | q_0.5 | ... | q_0.1 | q_0.5 | ... |
        pivot_table = df_pin.pivot(index='Model', columns=['Method', 'Metric'], values='Value')
        
        # Сортируем строки по сумме ошибок (или по медиане Vincentization)
        # Для простоты сортируем по индексу или можно посчитать среднее
        # pivot_table = pivot_table.sort_index()
        print("\n=== FINAL RESULTS (Pinball Loss per Quantile) ===")
        
    else:
        raise ValueError("metric_mode must be 'WIS' or 'Pinball'")

    # Стилизация (подсветка минимума в каждой строке)
    # Для MultiIndex 'Pinball' режима нам нужно подсвечивать минимум 
    # среди методов ДЛЯ КАЖДОГО квантиля отдельно.
    
    def highlight_min(s):
        # Если режим Pinball, у нас мультииндекс в колонках. 
        # Pandas apply(axis=1) идет по строкам. s - это строка.
        # Нам нужно сравнивать q_0.1 у Винцентизации с q_0.1 у Агрегатора.
        
        if isinstance(s.index, pd.MultiIndex):
            # Извлекаем уровни квантилей
            quantiles_levels = s.index.get_level_values(1).unique()
            is_min = pd.Series(False, index=s.index)
            
            for q_lvl in quantiles_levels:
                # Берем значения только этого квантиля для всех методов
                subset = s[:, q_lvl] 
                # Находим минимум
                min_val = subset.min()
                # Помечаем True те, что равны минимуму
                is_min.loc[:, q_lvl] = (subset == min_val)
                
            return ['background-color: #4ef048; font-weight: bold' if v else '' for v in is_min]
        else:
            # Обычный режим WIS
            is_min = s == s.min()
            return ['background-color: #4ef048; font-weight: bold' if v else '' for v in is_min]

    return pivot_table.style.apply(highlight_min, axis=1).format("{:.4f}")