import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_pinball_loss

def calculate_m5_scale(y_train):
    """
    Считает скейлер по методологии M5 (Mean Absolute Diff на трейне).
    Используется для нормализации ошибки разных рядов.
    """
    # Берем разности соседних элементов (y_t - y_{t-1})
    diffs = np.diff(y_train)
    # Считаем среднее абсолютное значение
    scale = np.mean(np.abs(diffs))
    
    # Защита от деления на ноль (для константных рядов)
    return scale if scale != 0 else 1.0

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
        scale = calculate_m5_scale(y_train)
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