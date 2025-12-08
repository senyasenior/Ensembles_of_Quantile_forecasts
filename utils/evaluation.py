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
    """
    n_samples = len(y_true)
    n_models = len(combo_names)
    n_quantiles = len(quantiles)
    
    # [Samples, Models, Quantiles]
    X_data = np.zeros((n_samples, n_models, n_quantiles))
    
    for m_idx, m_name in enumerate(combo_names):
        for q_idx, q in enumerate(quantiles):
            X_data[:, m_idx, q_idx] = models_dict[m_name][q]
            
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.float32).view(-1, 1)
    
    # Context (Series IDs)
    if series_ids is not None:
        context_tensor = torch.tensor(series_ids, dtype=torch.long)
    else:
        # Заглушка (Global mode)
        context_tensor = torch.zeros(n_samples, dtype=torch.long)
        
    return X_tensor, y_tensor, context_tensor

def train_and_eval_aggregator(
    X, y, context, 
    agg_config, 
    n_models, 
    quantiles, 
    device='cpu',
    epochs=30
):
    """
    Обучает агрегатор на Train-части OOF и оценивает на Test-части OOF.
    """
    # 1. Разбиение на Train/Test для агрегатора (например, 70/30)
    # Мы не можем учить агрегатор и проверять на одних данных (будет data leak)
    split_idx = int(len(y) * 0.7)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    c_train, c_test = context[:split_idx], context[split_idx:]
    
    # Датасеты
    # Trainer ждет (Inputs, Context, Target)
    train_ds = TensorDataset(X_train, c_train, y_train)
    # Для валидации используем тестовую часть
    val_ds = TensorDataset(X_test, c_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    
    # 2. Модель
    # Определяем параметры n_series для Embedding слоя, если нужно
    n_series_max = int(context.max().item()) + 1
    
    model = UniversalQuantileAggregator(
        n_models=n_models,
        quantiles=quantiles,
        weighting_type=agg_config['weighting_type'],
        resolution=agg_config['resolution'],
        n_series=n_series_max,
        # Для Local нужны features, тут пока заглушка, считаем что Series/Global
        dropout=0.1
    )
    
    # 3. Обучение
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = PinballLoss(quantiles)
    
    trainer = QuantileAggregatorTrainer(
        model, optimizer, criterion, torch.device(device), 
        checkpoint_dir='temp_checkpoints'
    )
    
    # Suppress output for loop
    # history = trainer.fit(train_loader, val_loader, epochs=epochs, early_stopping=5, verbose=False)
    # (Добавьте verbose=False в fit метод тренера, чтобы не спамить в консоль)
    trainer.fit(train_loader, val_loader, epochs=epochs, early_stopping=5)
    
    # 4. Предсказание на Test части
    model.eval()
    with torch.no_grad():
        final_preds = model.predict_ensemble(
            model_adapters=[], # Не используем адаптеры, у нас уже тензоры
            input_data=None, 
            context_input=None, # Передаем тензоры напрямую ниже
            horizon=0, # Заглушка
            post_sort=True 
        )
        
        # Хак: используем forward напрямую, так как у нас уже тензоры
        # predict_ensemble был для продакшна
        preds_tensor = model(X_test.to(device), c_test.to(device))
        # Post-sort
        preds_tensor, _ = torch.sort(preds_tensor, dim=1)
        final_preds = preds_tensor.cpu().numpy()
        
    return final_preds, y_test.numpy().flatten()

def evaluate_model_combinations_advanced(
    y_true, 
    models_dict, 
    quantiles, 
    y_train_hist=None, 
    aggregator_configs=[
        {'weighting_type': 'global', 'resolution': 'coarse'},
        {'weighting_type': 'global', 'resolution': 'medium'}
    ],
    series_ids=None,
    device='cpu'
):
    """
    Сравнивает модели, их комбинации (Винцентизация) и обучаемые Агрегаторы.
    
    Args:
        y_true: Истинные значения OOF.
        models_dict: Словарь OOF предсказаний.
        y_train_hist: История ряда (для расчета Scale).
        aggregator_configs: Список словарей с настройками агрегатора для перебора.
        series_ids: Массив ID рядов (для Series-Specific агрегации).
    """
    
    # 1. Расчет скейла
    scale = 1.0
    if y_train_hist is not None:
        scale = calculate_scale(y_train_hist)
        print(f"Scale computed on history: {scale:.4f}")
        
    model_names = list(models_dict.keys())
    results_list = [] # Будем собирать dict'и с результатами
    
    # Определяем срез для валидации (тот же 70/30, что внутри train_and_eval)
    # Нам нужно сравнивать Винцентизацию и Агрегатор НА ОДНИХ ДАННЫХ (Test часть)
    split_idx = int(len(y_true) * 0.7)
    y_true_test = y_true[split_idx:]
    
    # --- ЦИКЛ ПО КОМБИНАЦИЯМ ---
    for r in range(1, len(model_names) + 1):
        for combo in combinations(model_names, r):
            combo_name = " + ".join(combo)
            print(f"Evaluating: {combo_name}")
            
            # --- A. Vincentization (Baseline) ---
            # Собираем предикты (только на Test части, чтобы сравнение было честным)
            preds_np_list = []
            for q in quantiles:
                q_preds = []
                for m in combo:
                    # Берем срез [split_idx:]
                    q_preds.append(models_dict[m][q][split_idx:])
                # Среднее по моделям (shape: [N_test])
                avg_pred = np.mean(q_preds, axis=0)
                preds_np_list.append(avg_pred)
            
            # [N_test, n_quantiles]
            vinc_preds = np.stack(preds_np_list, axis=1)
            
            # Метрики Винцентизации
            metrics_v = evaluate_metrics(y_true_test, vinc_preds, quantiles, scale)
            results_list.append({
                'Model': combo_name,
                'Method': 'Vincentization',
                'WIS': metrics_v['WIS'],
                'MACE': metrics_v['MACE']
            })
            
            # --- B. Neural Aggregators (Только если > 1 модели) ---
            # Обучать агрегатор на 1 модели бессмысленно (он просто выучит identity)
            if len(combo) > 1:
                # Готовим тензоры (полные OOF)
                X_t, y_t, c_t = prepare_tensors_from_dict(
                    models_dict, combo, quantiles, y_true, series_ids
                )
                
                for conf in aggregator_configs:
                    agg_name = f"{conf['weighting_type']}-{conf['resolution']}"
                    
                    try:
                        agg_preds, agg_y_true = train_and_eval_aggregator(
                            X_t, y_t, c_t, conf, 
                            n_models=len(combo), 
                            quantiles=quantiles,
                            device=device
                        )
                        
                        # Метрики Агрегатора
                        # agg_y_true должен совпадать с y_true_test, но на всякий случай используем возвращенный
                        metrics_a = evaluate_metrics(agg_y_true, agg_preds, quantiles, scale)
                        
                        results_list.append({
                            'Model': combo_name,
                            'Method': f"Agg_{agg_name}",
                            'WIS': metrics_a['WIS'],
                            'MACE': metrics_a['MACE']
                        })
                    except Exception as e:
                        print(f"Error training {agg_name} for {combo_name}: {e}")

    # --- ФОРМИРОВАНИЕ ОТЧЕТА ---
    df_res = pd.DataFrame(results_list)
    
    # Пивот таблица: Строки - Комбинации, Столбцы - Методы, Значения - WIS
    pivot_wis = df_res.pivot(index='Model', columns='Method', values='WIS')
    
    # Сортировка
    pivot_wis = pivot_wis.sort_values(by='Vincentization')
    
    # Подсветка лучших
    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: #4ef048; font-weight: bold' if v else '' for v in is_min]
    
    return pivot_wis.style.apply(highlight_min, axis=1).format("{:.4f}")