#!/usr/bin/env python3
"""
Paper-Ready 벤치마크 분석 + 시각화 + LaTeX 테이블 생성

paper_benchmark.py의 출력 JSON을 분석하여:
  - 95% CI 포함 통계 테이블
  - matplotlib 시각화 (에러바, 박스플롯, 레이더, 파레토)
  - LaTeX 테이블 자동 생성
  - Kruskal-Wallis 통계 검정
  - 파레토 최적 분석

사용법:
    # 전체 분석 (테이블 + 차트 + LaTeX)
    python3 scripts/paper_benchmark_analysis.py \\
        --input ~/paper_benchmark_results/bench_maze_nav_20260317/aggregated_stats.json

    # LaTeX 테이블만
    python3 scripts/paper_benchmark_analysis.py --input stats.json --latex-only

    # 차트만
    python3 scripts/paper_benchmark_analysis.py --input stats.json --plot-only

    # 기존 controller_benchmark.py 결과 (단일 시행) 분석
    python3 scripts/paper_benchmark_analysis.py \\
        --input ~/benchmark_results/bench_20260317.json --legacy
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 메트릭 정의
# ============================================================

METRIC_DEFS = {
    # key: (display_name, unit, format, lower_is_better)
    'travel_time': ('Time', 's', '{:.1f}', True),
    'travel_distance': ('Distance', 'm', '{:.2f}', None),
    'position_error': ('Pos Error', 'm', '{:.3f}', True),
    'yaw_error': ('Yaw Error', 'rad', '{:.3f}', True),
    'mean_speed': ('Mean Speed', 'm/s', '{:.2f}', False),
    'max_speed': ('Max Speed', 'm/s', '{:.2f}', None),
    'rms_jerk': ('RMS Jerk', 'm/s^3', '{:.2f}', True),
    'mean_jerk_vx': ('Jerk Vx', 'm/s^3', '{:.2f}', True),
    'stall_ratio': ('Stall', '%', '{:.1%}', True),
    'min_obstacle_dist': ('Min Obs', 'm', '{:.2f}', False),
    'mean_obstacle_dist': ('Mean Obs', 'm', '{:.2f}', False),
    'near_misses': ('Near Miss', '', '{:.0f}', True),
    'collisions': ('Collisions', '', '{:.0f}', True),
}

# 논문 핵심 메트릭 (Table 1)
PAPER_METRICS = [
    'travel_time', 'position_error', 'rms_jerk',
    'min_obstacle_dist', 'collisions',
]

# 카테고리별 메트릭
CATEGORIES = {
    'Performance': ['travel_time', 'travel_distance', 'mean_speed'],
    'Accuracy': ['position_error', 'yaw_error'],
    'Smoothness': ['rms_jerk', 'mean_jerk_vx'],
    'Safety': ['min_obstacle_dist', 'mean_obstacle_dist',
               'near_misses', 'collisions'],
}


# ============================================================
# 데이터 로딩
# ============================================================

def load_stats(filepath: str, legacy: bool = False) -> Tuple[Dict, Dict, Dict]:
    """JSON 로드 → (metadata, config, stats)"""
    with open(filepath) as f:
        data = json.load(f)

    if legacy:
        # 기존 controller_benchmark.py 출력 변환
        return convert_legacy(data)

    return (
        data.get('metadata', {}),
        data.get('config', {}),
        data.get('stats', {}),
    )


def convert_legacy(data: Dict) -> Tuple[Dict, Dict, Dict]:
    """기존 controller_benchmark.py JSON → 신규 포맷 변환"""
    metadata = data.get('metadata', data.get('system_info', {}))
    config = data.get('config', {})

    stats = {}
    results = data.get('results', data.get('controllers', []))

    if isinstance(results, list):
        for r in results:
            ctrl = r.get('controller', r.get('name', 'unknown'))
            metrics = r.get('metrics', r)
            ctrl_stats = {
                'n_trials': 1,
                'n_success': 1 if metrics.get('goal_reached') else 0,
                'success_rate': 1.0 if metrics.get('goal_reached') else 0.0,
            }
            for key in METRIC_DEFS:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    v = float(metrics[key])
                    ctrl_stats[key] = {
                        'mean': v, 'std': 0, 'ci95': 0,
                        'min': v, 'max': v, 'values': [v],
                    }
            stats[ctrl] = ctrl_stats
    elif isinstance(results, dict):
        stats = results

    return metadata, config, stats


# ============================================================
# ASCII 테이블
# ============================================================

def print_summary_table(stats: Dict[str, Any], metrics: List[str] = None):
    """95% CI 포함 요약 테이블"""
    if metrics is None:
        metrics = PAPER_METRICS

    # 헤더
    header_parts = [f'{"Controller":<22s}', f'{"Succ":>5s}']
    for m in metrics:
        name = METRIC_DEFS.get(m, (m, '', '', None))[0]
        header_parts.append(f'{name:>16s}')

    print(f'\n{"=" * (28 + 16 * len(metrics))}')
    print('  ' + '  '.join(header_parts))
    print(f'  {"-" * (24 + 16 * len(metrics))}')

    # 각 컨트롤러
    for ctrl in sorted(stats.keys()):
        s = stats[ctrl]
        parts = [f'{ctrl:<22s}', f'{s["n_success"]}/{s["n_trials"]}']

        for m in metrics:
            v = s.get(m)
            if v is None:
                parts.append(f'{"---":>16s}')
                continue

            fmt = METRIC_DEFS.get(m, (m, '', '{:.2f}', None))[2]
            mean_str = fmt.format(v['mean'])
            if v['ci95'] > 0:
                ci_str = fmt.format(v['ci95'])
                parts.append(f'{mean_str}+/-{ci_str:>5s}')
            else:
                parts.append(f'{mean_str:>16s}')

        print('  ' + '  '.join(parts))

    print(f'{"=" * (28 + 16 * len(metrics))}\n')


def print_rankings(stats: Dict[str, Any]):
    """메트릭별 순위"""
    print(f'\n{"=" * 60}')
    print('  Rankings (by metric)')
    print(f'{"=" * 60}')

    for m in PAPER_METRICS:
        mdef = METRIC_DEFS.get(m, (m, '', '{:.2f}', None))
        name, _, fmt, lower_better = mdef

        vals = []
        for ctrl, s in stats.items():
            v = s.get(m)
            if v is not None:
                vals.append((ctrl, v['mean']))

        if not vals:
            continue

        reverse = not lower_better if lower_better is not None else False
        vals.sort(key=lambda x: x[1], reverse=reverse)

        best = vals[0]
        print(f'  {name:<16s}: 1st={best[0]} ({fmt.format(best[1])})')
        if len(vals) > 1:
            second = vals[1]
            print(f'  {"":16s}  2nd={second[0]} ({fmt.format(second[1])})')

    print()


# ============================================================
# 통계 검정
# ============================================================

def kruskal_wallis_test(stats: Dict[str, Any], metric: str) -> Optional[Dict]:
    """Kruskal-Wallis H-test (비모수 ANOVA 대체)"""
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return None

    groups = []
    labels = []
    for ctrl, s in stats.items():
        v = s.get(metric)
        if v is not None and len(v.get('values', [])) >= 2:
            groups.append(v['values'])
            labels.append(ctrl)

    if len(groups) < 2:
        return None

    try:
        h_stat, p_value = scipy_stats.kruskal(*groups)
        return {
            'test': 'Kruskal-Wallis',
            'metric': metric,
            'H': round(h_stat, 4),
            'p_value': round(p_value, 6),
            'significant': p_value < 0.05,
            'groups': labels,
            'n_groups': len(groups),
        }
    except Exception:
        return None


def run_statistical_tests(stats: Dict[str, Any]) -> List[Dict]:
    """모든 핵심 메트릭에 대해 통계 검정"""
    results = []
    for m in PAPER_METRICS:
        test = kruskal_wallis_test(stats, m)
        if test:
            results.append(test)
    return results


def print_statistical_tests(tests: List[Dict]):
    """통계 검정 결과 출력"""
    if not tests:
        print('\n  [scipy 미설치: 통계 검정 건너뜀]')
        return

    print(f'\n{"=" * 60}')
    print('  Statistical Tests (Kruskal-Wallis)')
    print(f'{"=" * 60}')
    for t in tests:
        sig = '*' if t['significant'] else ' '
        print(f'  {t["metric"]:<20s} H={t["H"]:.2f}  '
              f'p={t["p_value"]:.4f} {sig}  '
              f'(k={t["n_groups"]} groups)')
    print(f'  {"*"} p < 0.05')
    print()


# ============================================================
# 파레토 분석
# ============================================================

def pareto_frontier(stats: Dict[str, Any],
                    x_metric: str = 'travel_time',
                    y_metric: str = 'min_obstacle_dist') -> List[str]:
    """파레토 최적 컨트롤러 식별"""
    points = []
    for ctrl, s in stats.items():
        x_val = s.get(x_metric)
        y_val = s.get(y_metric)
        if x_val is not None and y_val is not None:
            points.append((ctrl, x_val['mean'], y_val['mean']))

    if not points:
        return []

    # x: lower is better (time), y: higher is better (obstacle dist)
    x_lower = METRIC_DEFS.get(x_metric, ('', '', '', True))[3]
    y_lower = METRIC_DEFS.get(y_metric, ('', '', '', False))[3]

    pareto = []
    for i, (ctrl_i, xi, yi) in enumerate(points):
        dominated = False
        for j, (ctrl_j, xj, yj) in enumerate(points):
            if i == j:
                continue
            # j가 i를 지배하는가?
            x_better = (xj < xi) if x_lower else (xj > xi)
            y_better = (yj > yi) if not y_lower else (yj < yi)
            x_eq = abs(xj - xi) < 1e-6
            y_eq = abs(yj - yi) < 1e-6

            if (x_better and (y_better or y_eq)) or \
               (y_better and (x_better or x_eq)):
                dominated = True
                break
        if not dominated:
            pareto.append(ctrl_i)

    return pareto


def print_pareto(stats: Dict[str, Any],
                 x_metric: str = 'travel_time',
                 y_metric: str = 'min_obstacle_dist'):
    """파레토 분석 결과 출력"""
    frontier = pareto_frontier(stats, x_metric, y_metric)
    x_name = METRIC_DEFS.get(x_metric, (x_metric,))[0]
    y_name = METRIC_DEFS.get(y_metric, (y_metric,))[0]

    print(f'\n{"=" * 60}')
    print(f'  Pareto Frontier ({x_name} vs {y_name})')
    print(f'{"=" * 60}')

    for ctrl, s in sorted(stats.items()):
        x_val = s.get(x_metric)
        y_val = s.get(y_metric)
        if x_val is None or y_val is None:
            continue
        is_pareto = ctrl in frontier
        marker = ' *' if is_pareto else '  '
        print(f' {marker} {ctrl:<20s} {x_name}={x_val["mean"]:.2f}  '
              f'{y_name}={y_val["mean"]:.2f}')

    print(f'\n  * = Pareto-optimal ({len(frontier)} controllers)')
    print()


# ============================================================
# LaTeX 테이블 생성
# ============================================================

def generate_latex_table(stats: Dict[str, Any],
                         metrics: List[str] = None,
                         caption: str = 'Navigation Performance Comparison',
                         label: str = 'tab:performance') -> str:
    """LaTeX 테이블 생성 (mean +/- CI, 최적값 볼드)"""
    if metrics is None:
        metrics = PAPER_METRICS

    # 각 메트릭별 최적값 찾기
    best_vals = {}
    for m in metrics:
        mdef = METRIC_DEFS.get(m, (m, '', '{:.2f}', None))
        lower_better = mdef[3]
        if lower_better is None:
            continue

        vals = [(ctrl, s.get(m, {}).get('mean', float('inf') if lower_better else -float('inf')))
                for ctrl, s in stats.items()
                if s.get(m) is not None]
        if vals:
            if lower_better:
                best_ctrl = min(vals, key=lambda x: x[1])[0]
            else:
                best_ctrl = max(vals, key=lambda x: x[1])[0]
            best_vals[m] = best_ctrl

    # LaTeX 생성
    n_cols = 2 + len(metrics)  # Controller + Success + metrics
    col_spec = 'l' + 'c' * (n_cols - 1)

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append(r'\small')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')

    # 헤더
    headers = ['Controller', 'Success']
    for m in metrics:
        mdef = METRIC_DEFS.get(m, (m, '', '', None))
        name = mdef[0]
        unit = mdef[1]
        if unit:
            headers.append(f'{name} ({unit})')
        else:
            headers.append(name)
    lines.append(' & '.join(headers) + r' \\')
    lines.append(r'\midrule')

    # 데이터 행
    for ctrl in sorted(stats.keys()):
        s = stats[ctrl]
        row = [ctrl.replace('_', r'\_')]
        row.append(f'{s["n_success"]}/{s["n_trials"]}')

        for m in metrics:
            v = s.get(m)
            if v is None:
                row.append('---')
                continue

            fmt = METRIC_DEFS.get(m, (m, '', '{:.2f}', None))[2]
            mean_str = fmt.format(v['mean'])

            if v['ci95'] > 0:
                ci_str = fmt.format(v['ci95'])
                cell = f'${mean_str} \\pm {ci_str}$'
            else:
                cell = f'${mean_str}$'

            # 최적값 볼드
            if best_vals.get(m) == ctrl:
                cell = r'\textbf{' + cell + '}'

            row.append(cell)

        lines.append(' & '.join(row) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_latex_safety_table(stats: Dict[str, Any]) -> str:
    """안전성 비교 LaTeX 테이블"""
    safety_metrics = ['travel_time', 'min_obstacle_dist',
                      'mean_obstacle_dist', 'near_misses', 'collisions']
    return generate_latex_table(
        stats, safety_metrics,
        caption='Safety-Augmented Controller Comparison',
        label='tab:safety')


# ============================================================
# matplotlib 시각화
# ============================================================

def plot_error_bars(stats: Dict[str, Any], output_dir: Path):
    """에러바 차트 (mean + 95% CI)"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print('  [matplotlib 미설치: 차트 건너뜀]')
        return

    metrics_to_plot = ['travel_time', 'position_error', 'rms_jerk',
                       'min_obstacle_dist', 'mean_speed', 'collisions']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('MPPI Controller Benchmark (mean +/- 95% CI)', fontsize=14)

    controllers = sorted(stats.keys())

    for idx, m in enumerate(metrics_to_plot):
        ax = axes[idx // 3][idx % 3]
        mdef = METRIC_DEFS.get(m, (m, '', '{:.2f}', None))
        name, unit, _, lower_better = mdef

        means = []
        ci95s = []
        colors = []
        for ctrl in controllers:
            v = stats[ctrl].get(m)
            if v is not None:
                means.append(v['mean'])
                ci95s.append(v['ci95'])
            else:
                means.append(0)
                ci95s.append(0)

        # 색상: 최적값 강조
        if lower_better is not None and means:
            if lower_better:
                best_idx = means.index(min(m for m in means if m > 0) if any(m > 0 for m in means) else 0)
            else:
                best_idx = means.index(max(means))
            colors = ['#2196F3' if i != best_idx else '#4CAF50'
                      for i in range(len(controllers))]
        else:
            colors = ['#2196F3'] * len(controllers)

        x = range(len(controllers))
        bars = ax.bar(x, means, yerr=ci95s, capsize=3,
                      color=colors, alpha=0.8, edgecolor='white')
        ax.set_ylabel(f'{name} ({unit})' if unit else name)
        ax.set_xticks(x)
        ax.set_xticklabels([c[:10] for c in controllers],
                           rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'error_bar_chart.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Chart: {plot_path}')


def plot_box_plots(stats: Dict[str, Any], output_dir: Path):
    """박스플롯 (다중 시행 분포)"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        return

    metrics_to_plot = ['travel_time', 'position_error', 'rms_jerk',
                       'min_obstacle_dist']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution of Metrics Across Trials', fontsize=14)

    controllers = sorted(stats.keys())

    for idx, m in enumerate(metrics_to_plot):
        ax = axes[idx // 2][idx % 2]
        mdef = METRIC_DEFS.get(m, (m, '', '{:.2f}', None))
        name, unit = mdef[0], mdef[1]

        box_data = []
        box_labels = []
        for ctrl in controllers:
            v = stats[ctrl].get(m)
            if v is not None and len(v.get('values', [])) > 1:
                box_data.append(v['values'])
                box_labels.append(ctrl[:12])

        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#42A5F5')
                patch.set_alpha(0.7)
            ax.set_ylabel(f'{name} ({unit})' if unit else name)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'box_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Chart: {plot_path}')


def plot_radar(stats: Dict[str, Any], output_dir: Path):
    """레이더 차트 (정규화 비교)"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        matplotlib.use('Agg')
    except ImportError:
        return

    radar_metrics = ['travel_time', 'position_error', 'rms_jerk',
                     'min_obstacle_dist', 'mean_speed']
    controllers = sorted(stats.keys())[:8]  # 상위 8개

    # 정규화 (0-1)
    normalized = {}
    for m in radar_metrics:
        vals = [stats[c].get(m, {}).get('mean', 0)
                for c in controllers]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax > vmin else 1.0
        lower_better = METRIC_DEFS.get(m, ('', '', '', None))[3]

        norm_vals = []
        for v in vals:
            n = (v - vmin) / rng
            if lower_better:
                n = 1.0 - n  # 낮을수록 좋으면 반전
            norm_vals.append(n)
        normalized[m] = norm_vals

    # 레이더 플롯
    N = len(radar_metrics)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_title('Normalized Controller Comparison\n(outer = better)',
                 fontsize=14, pad=20)

    colors = plt.cm.Set2(np.linspace(0, 1, len(controllers)))

    for i, ctrl in enumerate(controllers):
        values = [normalized[m][i] for m in radar_metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=ctrl[:15],
                color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_DEFS.get(m, (m,))[0]
                        for m in radar_metrics])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plot_path = output_dir / 'radar_chart.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Chart: {plot_path}')


def plot_pareto(stats: Dict[str, Any], output_dir: Path,
                x_metric: str = 'travel_time',
                y_metric: str = 'min_obstacle_dist'):
    """파레토 프론티어 차트"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        return

    frontier = pareto_frontier(stats, x_metric, y_metric)

    fig, ax = plt.subplots(figsize=(10, 8))
    x_name = METRIC_DEFS.get(x_metric, (x_metric,))[0]
    y_name = METRIC_DEFS.get(y_metric, (y_metric,))[0]
    ax.set_title(f'Pareto Frontier: {x_name} vs {y_name}', fontsize=14)

    for ctrl, s in stats.items():
        x_val = s.get(x_metric)
        y_val = s.get(y_metric)
        if x_val is None or y_val is None:
            continue

        is_pareto = ctrl in frontier
        color = '#4CAF50' if is_pareto else '#90A4AE'
        marker = 's' if is_pareto else 'o'
        size = 120 if is_pareto else 60

        ax.scatter(x_val['mean'], y_val['mean'],
                   s=size, c=color, marker=marker, zorder=3,
                   edgecolors='white', linewidths=1.5)

        # 에러바
        if x_val['ci95'] > 0 or y_val['ci95'] > 0:
            ax.errorbar(x_val['mean'], y_val['mean'],
                        xerr=x_val['ci95'], yerr=y_val['ci95'],
                        fmt='none', ecolor=color, alpha=0.4, capsize=3)

        ax.annotate(ctrl[:12], (x_val['mean'], y_val['mean']),
                    textcoords='offset points', xytext=(5, 5),
                    fontsize=8, alpha=0.8)

    x_unit = METRIC_DEFS.get(x_metric, ('', ''))[1]
    y_unit = METRIC_DEFS.get(y_metric, ('', ''))[1]
    ax.set_xlabel(f'{x_name} ({x_unit})', fontsize=12)
    ax.set_ylabel(f'{y_name} ({y_unit})', fontsize=12)
    ax.grid(alpha=0.3)

    # 범례
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4CAF50',
               markersize=12, label='Pareto-optimal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#90A4AE',
               markersize=10, label='Dominated'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plot_path = output_dir / 'pareto_frontier.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Chart: {plot_path}')


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Paper-Ready Benchmark Analysis + Visualization')
    parser.add_argument(
        '--input', type=str, required=True,
        help='aggregated_stats.json 또는 legacy benchmark JSON')
    parser.add_argument(
        '--legacy', action='store_true',
        help='기존 controller_benchmark.py JSON 포맷')
    parser.add_argument(
        '--latex-only', action='store_true',
        help='LaTeX 테이블만 생성')
    parser.add_argument(
        '--plot-only', action='store_true',
        help='차트만 생성')
    parser.add_argument(
        '--output-dir', type=str, default='',
        help='출력 디렉토리 (기본: 입력 파일과 같은 위치)')
    parser.add_argument(
        '--metrics', type=str, default='',
        help='표시할 메트릭 (쉼표 구분)')

    args = parser.parse_args()

    # 데이터 로드
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f'[ERROR] File not found: {args.input}')
        sys.exit(1)

    metadata, config, stats = load_stats(str(input_path), args.legacy)

    if not stats:
        print('[ERROR] No stats data found')
        sys.exit(1)

    # 출력 디렉토리
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 메트릭 선택
    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',')]

    # 메타데이터 출력
    print(f'\n  Paper Benchmark Analysis')
    print(f'  Input: {input_path}')
    if metadata:
        print(f'  Git: {metadata.get("git_hash", "?")} '
              f'ROS: {metadata.get("ros_distro", "?")} '
              f'CPU: {metadata.get("cpu_model", "?")[:40]}')
    print(f'  Controllers: {len(stats)}')
    print()

    if args.latex_only:
        # LaTeX만
        latex = generate_latex_table(stats, metrics)
        print(latex)
        latex_path = output_dir / 'table_performance.tex'
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f'\n  LaTeX: {latex_path}')

        latex_safety = generate_latex_safety_table(stats)
        safety_path = output_dir / 'table_safety.tex'
        with open(safety_path, 'w') as f:
            f.write(latex_safety)
        print(f'  LaTeX: {safety_path}')
        return

    if args.plot_only:
        # 차트만
        plot_error_bars(stats, output_dir)
        plot_box_plots(stats, output_dir)
        plot_radar(stats, output_dir)
        plot_pareto(stats, output_dir)
        return

    # 전체 분석
    # 1. 요약 테이블
    print_summary_table(stats, metrics)

    # 2. 순위
    print_rankings(stats)

    # 3. 통계 검정
    tests = run_statistical_tests(stats)
    print_statistical_tests(tests)

    # 4. 파레토 분석
    print_pareto(stats)

    # 5. LaTeX 테이블
    latex = generate_latex_table(stats, metrics)
    latex_path = output_dir / 'table_performance.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f'  LaTeX: {latex_path}')

    latex_safety = generate_latex_safety_table(stats)
    safety_path = output_dir / 'table_safety.tex'
    with open(safety_path, 'w') as f:
        f.write(latex_safety)
    print(f'  LaTeX: {safety_path}')

    # 6. 차트
    plot_error_bars(stats, output_dir)
    plot_box_plots(stats, output_dir)
    plot_radar(stats, output_dir)
    plot_pareto(stats, output_dir)

    # 7. 통계 검정 결과 저장
    if tests:
        test_path = output_dir / 'statistical_tests.json'
        with open(test_path, 'w') as f:
            json.dump(tests, f, indent=2, default=str)
        print(f'  Tests: {test_path}')

    print(f'\n  Analysis complete. Output: {output_dir}')


if __name__ == '__main__':
    main()
