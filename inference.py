import pickle
import os
import re
import numpy as np
import pandas as pd

BASE_DIR = 'data/input.csv'  # заменить на реальный путь при необходимости
MODELS_DIR = os.path.join(BASE_DIR, '')  # заменить на реальный путь при необходимости
DATA_DIR = os.path.join(BASE_DIR, '')  # заменить на реальный путь при необходимости

L_MIN, L_MAX = 79.5, 100.0
A_MIN, A_MAX = -1.0, -0.2
B_MIN, B_MAX = -4.0, -2.5

TARGET_MAE = {
    'L': 0.00,
    'a': 0.11,
    'b': 0.29,
}

TARGET_BOUNDS = {
    'L': (L_MIN, L_MAX),
    'a': (A_MIN, A_MAX),
    'b': (B_MIN, B_MAX),
}

_ID_COLS = ['vivilen_num', 'Номер партии']
_LAB_COLS = ['L', 'a', 'b']
LAG_PERIODS = [1, 2, 3]
ROLLING_WINDOWS = [3]
_TOP_NUM_FEATURES_FOR_LAGS = 5
_USE_LAB_LAGS = True
_INCLUDE_TARGET_LAGS = True

MIN_ROWS_REQUIRED = 3


# Вспомогательные функции
# =========================================================

def _sanitize_feature_name(name):
    name = str(name)
    name = re.sub(r'[$$\<\>\{\}\"\':\,\s\\/]+', '_', name)
    name = re.sub(r'[^0-9a-zA-Zа-яА-Я_\.]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if name == '':
        name = 'feature'
    return name


def _sanitize_dataframe_columns(df):
    df = df.copy()
    cleaned = [_sanitize_feature_name(col) for col in df.columns]
    counts, final_cols = {}, []
    for col in cleaned:
        if col not in counts:
            counts[col] = 0
            final_cols.append(col)
        else:
            counts[col] += 1
            final_cols.append(f"{col}_{counts[col]}")
    df.columns = final_cols
    return df


def _add_date_features(df, date_col):
    df = df.copy()
    if date_col is None or date_col not in df.columns:
        return df
    dt = pd.to_datetime(df[date_col], errors='coerce')
    df[f'{date_col}_year'] = dt.dt.year
    df[f'{date_col}_month'] = dt.dt.month
    df[f'{date_col}_quarter'] = dt.dt.quarter
    df[f'{date_col}_weekofyear'] = dt.dt.isocalendar().week.astype('float')
    df[f'{date_col}_day'] = dt.dt.day
    df[f'{date_col}_dayofweek'] = dt.dt.dayofweek
    df[f'{date_col}_is_month_start'] = dt.dt.is_month_start.astype('float')
    df[f'{date_col}_is_month_end'] = dt.dt.is_month_end.astype('float')
    df[f'{date_col}_time_index'] = np.arange(len(df), dtype=np.float32)
    df[f'{date_col}_month_sin'] = np.sin(2 * np.pi * df[f'{date_col}_month'] / 12.0)
    df[f'{date_col}_month_cos'] = np.cos(2 * np.pi * df[f'{date_col}_month'] / 12.0)
    df[f'{date_col}_dow_sin'] = np.sin(2 * np.pi * df[f'{date_col}_dayofweek'] / 7.0)
    df[f'{date_col}_dow_cos'] = np.cos(2 * np.pi * df[f'{date_col}_dayofweek'] / 7.0)
    return df


def _get_lab_lag_sources(df, target, lab_cols=('L', 'a', 'b'),
                         include_target_lags=True):
    cols = []
    for c in lab_cols:
        if c in df.columns:
            if c == target and not include_target_lags:
                continue
            cols.append(c)
    return cols


def _select_top_numeric_features_for_lags(df, target, id_cols=None, top_n=5):
    if id_cols is None:
        id_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(id_cols + [target])
    for t in ['L', 'a', 'b']:
        if t in df.columns:
            exclude.add(t)
    candidate_cols = [c for c in numeric_cols if c not in exclude]
    if len(candidate_cols) == 0:
        return []
    if top_n is None or top_n >= len(candidate_cols):
        return candidate_cols
    tmp = df[candidate_cols + [target]].copy()
    corr = (tmp.corr(numeric_only=True)[target]
            .drop(target, errors='ignore')
            .abs()
            .sort_values(ascending=False))
    return corr.head(top_n).index.tolist()


def _add_lag_features(df, feature_cols, lag_periods=(1, 2), add_delta=True):
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        for lag in lag_periods:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        if add_delta and 1 in lag_periods and 2 in lag_periods:
            df[f'{col}_delta_1'] = df[col].shift(1) - df[col].shift(2)
    return df


def _add_rolling_features(df, feature_cols, rolling_windows=(3,)):
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        shifted = df[col].shift(1)
        for win in rolling_windows:
            df[f'{col}_roll_mean_{win}'] = shifted.rolling(window=win, min_periods=win).mean()
            df[f'{col}_roll_std_{win}'] = shifted.rolling(window=win, min_periods=win).std()
    return df


def _build_feature_set(df, target, feature_mode,
                       id_cols=None, date_col=None,
                       top_num_features_for_lags=5,
                       lag_periods=(1, 2, 3),
                       rolling_windows=(3,),
                       use_lab_lags=True,
                       lab_cols=('L', 'a', 'b'),
                       include_target_lags=True,
                       locked_lag_cols=None):  # ← НОВЫЙ ПАРАМЕТР
    """
    locked_lag_cols : dict или None
        Если передан — используем зафиксированные при обучении колонки
        вместо пересчёта через _select_top_numeric_features_for_lags.
        Ожидаемый формат:
            {
                'lab_lag_feature_cols':  [...],
                'auto_lag_feature_cols': [...],
                'all_lag_feature_cols':  [...],
            }
    """
    if id_cols is None:
        id_cols = []

    work_df = df.copy()

    if date_col is not None and date_col in work_df.columns:
        work_df = work_df.sort_values(date_col).reset_index(drop=True)
    else:
        work_df = work_df.reset_index(drop=True)

    auto_lag_feature_cols = []
    lab_lag_feature_cols = []
    all_lag_feature_cols = []

    if feature_mode in ['date', 'date_lag', 'date_lag_roll']:
        work_df = _add_date_features(work_df, date_col)

    if feature_mode in ['date_lag', 'date_lag_roll']:

        # ── lab-лаги ──────────────────────────────────────────────────────
        if use_lab_lags:
            if locked_lag_cols is not None:
                # инференс: берём зафиксированный список
                lab_lag_feature_cols = locked_lag_cols['lab_lag_feature_cols']
            else:
                # обучение: вычисляем
                lab_lag_feature_cols = _get_lab_lag_sources(
                    df=work_df, target=target,
                    lab_cols=lab_cols,
                    include_target_lags=include_target_lags
                )

            # фильтруем: оставляем только те, что реально есть в work_df
            lab_lag_feature_cols = [
                c for c in lab_lag_feature_cols if c in work_df.columns
            ]
            work_df = _add_lag_features(
                work_df, feature_cols=lab_lag_feature_cols,
                lag_periods=lag_periods, add_delta=True
            )

        # ── auto-лаги ─────────────────────────────────────────────────────
        if locked_lag_cols is not None:
            # инференс: берём зафиксированный список
            auto_lag_feature_cols = locked_lag_cols['auto_lag_feature_cols']
        else:
            # обучение: вычисляем
            auto_lag_feature_cols = _select_top_numeric_features_for_lags(
                df=work_df, target=target,
                id_cols=id_cols, top_n=top_num_features_for_lags
            )

        # фильтруем: оставляем только те, что реально есть в work_df
        auto_lag_feature_cols = [
            c for c in auto_lag_feature_cols if c in work_df.columns
        ]
        work_df = _add_lag_features(
            work_df, feature_cols=auto_lag_feature_cols,
            lag_periods=lag_periods, add_delta=True
        )

        # ── объединяем без дублей ──────────────────────────────────────────
        if locked_lag_cols is not None:
            all_lag_feature_cols = locked_lag_cols['all_lag_feature_cols']
        else:
            seen = set()
            for c in lab_lag_feature_cols + auto_lag_feature_cols:
                if c not in seen:
                    seen.add(c)
                    all_lag_feature_cols.append(c)

    if feature_mode == 'date_lag_roll':
        work_df = _add_rolling_features(
            work_df, feature_cols=all_lag_feature_cols,
            rolling_windows=rolling_windows
        )

    drop_cols = set(id_cols + [target])
    for t in ['L', 'a', 'b']:
        if t != target and t in work_df.columns:
            drop_cols.add(t)
    if date_col is not None and date_col in work_df.columns:
        drop_cols.add(date_col)

    X = work_df.drop(
        columns=[c for c in drop_cols if c in work_df.columns],
        errors='ignore'
    ).copy()
    y = work_df[target].copy()

    X = X.select_dtypes(include=[np.number]).copy()

    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    valid_mask = X.notna().all(axis=1)
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    X = _sanitize_dataframe_columns(X)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    meta = {
        'lab_lag_feature_cols': lab_lag_feature_cols,
        'auto_lag_feature_cols': auto_lag_feature_cols,
        'all_lag_feature_cols': all_lag_feature_cols,
    }

    return X.reset_index(drop=True), y.reset_index(drop=True), meta


# Загрузка моделей
# =========================================================

def load_saved_models(save_dir=MODELS_DIR):
    models = {}
    for tgt in ['L', 'a', 'b']:
        path = os.path.join(save_dir, f'best_model_target_{tgt}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Не найден файл модели: {path}\n"
                f"Проверь директорию: {save_dir}"
            )
        with open(path, 'rb') as f:
            models[tgt] = pickle.load(f)
        print(f"  Загружена модель: target={tgt} | "
              f"model_type={models[tgt]['model_type']} | "
              f"feature_mode={models[tgt]['feature_mode']} | "
              f"n_features={len(models[tgt]['feature_cols'])}")
    return models


# Загрузка данных
# =========================================================

def load_data(data_dir=DATA_DIR):
    """
    Ищет CSV файл в DATA_DIR.
    Если файлов несколько — берёт первый найденный.
    """
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"CSV файлы не найдены в директории: {data_dir}"
        )

    if len(csv_files) > 1:
        print(f"  [INFO] Найдено несколько CSV файлов: {csv_files}")
        print(f"  [INFO] Используем первый: {csv_files[0]}")

    csv_path = os.path.join(data_dir, csv_files[0])
    df = pd.read_csv(csv_path)
    print(f"  Загружен файл : {csv_path}")
    print(f"  Строк: {len(df)}, Колонок: {len(df.columns)}")
    print(f"  Колонки: {df.columns.tolist()}")
    return df


# Классификатор зон
# =========================================================

def classify_prediction(pred, mae, tgt_min, tgt_max):
    in_range = tgt_min <= pred <= tgt_max

    if not in_range:
        return 'red', (
            f"КРАСНАЯ: прогнозное значение {pred:.4f} вне допуска "
            f"[{tgt_min}, {tgt_max}]"
        )

    lower = pred - mae
    upper = pred + mae
    interval_ok = (lower >= tgt_min) and (upper <= tgt_max)

    if interval_ok:
        return 'green', (
                f"ЗЕЛЁНАЯ: прогнозное значение {pred:.4f} в допуске [{tgt_min}, {tgt_max}]"
                + (f", интервал [{lower:.4f}, {upper:.4f}] тоже в допуске"
                   if mae > 0 else "")
        )
    else:
        return 'yellow', (
            f"ЖЁЛТАЯ: прогнозное значение {pred:.4f} в допуске [{tgt_min}, {tgt_max}], "
            f"но интервал [{lower:.4f}, {upper:.4f}] частично выходит за границы"
        )


# Инференс для одного таргета
# =========================================================

def predict_single_target(df_input, target, model_info):
    feature_mode = model_info['feature_mode']
    date_col = model_info['date_col']
    feature_cols = model_info['feature_cols']
    model = model_info['model']

    lag_periods = model_info['lag_periods']
    rolling_windows = model_info['rolling_windows']

    # ← достаём зафиксированные колонки из pickle
    locked_lag_cols = {
        'lab_lag_feature_cols': model_info.get('lab_lag_feature_cols', []),
        'auto_lag_feature_cols': model_info.get('auto_lag_feature_cols', []),
        'all_lag_feature_cols': model_info.get('all_lag_feature_cols', []),
    }
    # если все три пустые — значит старый pickle без meta, fallback к None
    if not any(locked_lag_cols.values()):
        locked_lag_cols = None

    X_inf, _, _ = _build_feature_set(
        df=df_input,
        target=target,
        feature_mode=feature_mode,
        id_cols=_ID_COLS,
        date_col=date_col,
        top_num_features_for_lags=_TOP_NUM_FEATURES_FOR_LAGS,
        lag_periods=LAG_PERIODS,
        rolling_windows=ROLLING_WINDOWS,
        use_lab_lags=_USE_LAB_LAGS,
        lab_cols=_LAB_COLS,
        include_target_lags=_INCLUDE_TARGET_LAGS,
        locked_lag_cols=locked_lag_cols
    )

    if len(X_inf) == 0:
        raise ValueError(
            f"После построения признаков не осталось строк для таргета '{target}'. "
            f"Нужно минимум {MIN_ROWS_REQUIRED} строк."
        )

    # Выравниваем колонки по обучению
    missing_cols = [c for c in feature_cols if c not in X_inf.columns]
    extra_cols = [c for c in X_inf.columns if c not in feature_cols]

    if missing_cols:
        print(f"  [WARN] target={target}: отсутствуют колонки {missing_cols}, "
              f"заполняем нулями")
        for c in missing_cols:
            X_inf[c] = 0.0

    if extra_cols:
        print(f"  [WARN] target={target}: лишние колонки {extra_cols}, удаляем")
        X_inf = X_inf.drop(columns=extra_cols)

    X_inf = X_inf[feature_cols].astype(np.float32)

    pred = float(model.predict(X_inf.iloc[[-1]])[0])
    return pred


# Основная функция инференса
# =========================================================

def run_inference(df_input, saved_models):
    if len(df_input) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Слишком мало строк: {len(df_input)}. "
            f"Нужно минимум {MIN_ROWS_REQUIRED}."
        )

    results = {}

    for target in ['L', 'a', 'b']:
        print(f"\n  --- Таргет: {target} ---")
        try:
            pred = predict_single_target(df_input, target, saved_models[target])
            mae = TARGET_MAE[target]
            tgt_min, tgt_max = TARGET_BOUNDS[target]

            zone, description = classify_prediction(pred, mae, tgt_min, tgt_max)

            results[target] = {
                'prediction': pred,
                'mae': mae,
                'zone': zone,
                'description': description,
            }
            print(f"  Прогноз : {pred:.4f}")
            print(f"  MAE     : {mae:.6f}")
            print(f"  Зона    : {description}")

        except Exception as e:
            print(f"  [ERROR] target={target}: {e}")
            results[target] = {
                'prediction': None,
                'mae': None,
                'zone': 'error',
                'description': str(e),
            }

    return results


# Запуск
# =========================================================

print("=" * 60)
print("Загружаем данные...")
df_new = load_data(data_dir=DATA_DIR)

print("\nЗагружаем модели...")
saved_models = load_saved_models(save_dir=MODELS_DIR)

print("\nЗапускаем инференс...")
results = run_inference(df_new, saved_models)

print("Итоговый результат")
print("=" * 60)
for tgt, res in results.items():
    print(f"  {tgt}: {res['description']}")
