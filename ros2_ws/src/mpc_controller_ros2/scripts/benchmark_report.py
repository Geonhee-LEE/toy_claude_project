#!/usr/bin/env python3
"""
benchmark_report.py
===================
벤치마크 JSON 결과 → ASCII 비교 테이블 + CSV + 선택적 matplotlib 플롯

사용법:
    # 단일 결과 리포트
    python3 benchmark_report.py ~/benchmark_results/bench_20260316_120000.json

    # 복수 결과 비교 (서로 다른 world/설정 비교)
    python3 benchmark_report.py result1.json result2.json

    # CSV만 내보내기
    python3 benchmark_report.py result.json --csv-only

    # matplotlib 플롯 생성
    python3 benchmark_report.py result.json --plot

    # 특정 메트릭으로 정렬
    python3 benchmark_report.py result.json --sort-by rms_jerk

    # 랭킹 테이블
    python3 benchmark_report.py result.json --ranking
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── 메트릭 정의 ─────────────────────────────────────────────────

METRIC_DEFS = {
    # (display_name, unit, format, lower_is_better)
    'goal_reached':       ('Goal',      '',    '{!s:>5}',  None),
    'travel_time':        ('Time',      's',   '{:.1f}',   True),
    'travel_distance':    ('Dist',      'm',   '{:.2f}',   None),  # 상황 의존
    'position_error':     ('PosErr',    'm',   '{:.3f}',   True),
    'yaw_error':          ('YawErr',    'rad', '{:.3f}',   True),
    'mean_speed':         ('MeanSpd',   'm/s', '{:.2f}',   None),
    'max_speed':          ('MaxSpd',    'm/s', '{:.2f}',   None),
    'mean_jerk_vx':       ('JerkVx',    'm/s³','{:.2f}',   True),
    'mean_jerk_vy':       ('JerkVy',    'm/s³','{:.2f}',   True),
    'mean_jerk_omega':    ('JerkOm',    'r/s³','{:.2f}',   True),
    'rms_jerk':           ('RMSJerk',   'm/s³','{:.2f}',   True),
    'stall_ratio':        ('Stall',     '%',   '{:.1%}',   True),
    'min_obstacle_dist':  ('MinObs',    'm',   '{:.2f}',   False),  # 높을수록 안전
    'mean_obstacle_dist': ('MeanObs',   'm',   '{:.2f}',   False),
    'near_misses':        ('NearMiss',  '',    '{:d}',     True),
    'collisions':         ('Collisions','',    '{:d}',     True),
}

# 요약 테이블에 표시할 핵심 메트릭
SUMMARY_METRICS = [
    'goal_reached', 'travel_time', 'position_error',
    'rms_jerk', 'min_obstacle_dist', 'collisions', 'mean_speed',
]

# 상세 테이블 메트릭
DETAIL_METRICS = list(METRIC_DEFS.keys())

# 카테고리 분류
METRIC_CATEGORIES = {
    'Performance': ['travel_time', 'travel_distance', 'mean_speed', 'max_speed'],
    'Accuracy': ['position_error', 'yaw_error'],
    'Smoothness': ['mean_jerk_vx', 'mean_jerk_vy', 'mean_jerk_omega', 'rms_jerk'],
    'Safety': ['min_obstacle_dist', 'mean_obstacle_dist', 'near_misses', 'collisions'],
    'Efficiency': ['stall_ratio'],
}


def load_results(filepath: str) -> Dict:
    """JSON 결과 파일 로드"""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_successful_results(data: Dict) -> List[Dict]:
    """성공한 결과만 필터링"""
    results = data.get('results', [])
    return [r for r in results if r.get('status') == 'SUCCESS' and r.get('metrics')]


def format_metric(key: str, value: Any) -> str:
    """메트릭 값을 형식화된 문자열로 변환"""
    if value is None or value == '':
        return '—'

    _, _, fmt, _ = METRIC_DEFS.get(key, ('', '', '{!s}', None))

    try:
        if key == 'goal_reached':
            return '✓' if value else '✗'
        if key == 'min_obstacle_dist' and value == float('inf'):
            return '∞'
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def compute_rankings(results: List[Dict], metric_keys: List[str]) -> Dict[str, List[Tuple[str, Any]]]:
    """각 메트릭별 순위 계산"""
    rankings = {}

    for key in metric_keys:
        _, _, _, lower_is_better = METRIC_DEFS.get(key, ('', '', '{}', None))
        if lower_is_better is None:
            continue

        scored = []
        for r in results:
            metrics = r.get('metrics', {})
            val = metrics.get(key)
            if val is not None and val != '' and val != float('inf'):
                scored.append((r['controller'], val))

        if not scored:
            continue

        scored.sort(key=lambda x: x[1], reverse=not lower_is_better)
        rankings[key] = scored

    return rankings


def print_summary_table(results: List[Dict], sort_by: Optional[str] = None):
    """요약 비교 테이블 출력"""
    if not results:
        print('  No successful results to display.')
        return

    # 정렬
    if sort_by and sort_by in METRIC_DEFS:
        _, _, _, lower_is_better = METRIC_DEFS[sort_by]
        results = sorted(
            results,
            key=lambda r: r.get('metrics', {}).get(sort_by, float('inf')
                if lower_is_better else float('-inf')),
            reverse=not (lower_is_better if lower_is_better is not None else True)
        )

    # 컬럼 폭 계산
    ctrl_width = max(len(r['controller']) for r in results) + 1
    ctrl_width = max(ctrl_width, 12)

    # 헤더
    header_parts = [f'{"Controller":<{ctrl_width}}']
    for key in SUMMARY_METRICS:
        display_name, unit, _, _ = METRIC_DEFS[key]
        col_header = display_name
        header_parts.append(f'{col_header:>10}')
    header = ' '.join(header_parts)

    print()
    print('┌' + '─' * (len(header) + 2) + '┐')
    print('│ ' + header + ' │')
    print('├' + '─' * (len(header) + 2) + '┤')

    for r in results:
        metrics = r.get('metrics', {})
        parts = [f'{r["controller"]:<{ctrl_width}}']
        for key in SUMMARY_METRICS:
            val = metrics.get(key)
            parts.append(f'{format_metric(key, val):>10}')
        print('│ ' + ' '.join(parts) + ' │')

    print('└' + '─' * (len(header) + 2) + '┘')


def print_category_tables(results: List[Dict]):
    """카테고리별 상세 테이블 출력"""
    if not results:
        return

    ctrl_width = max(len(r['controller']) for r in results) + 1
    ctrl_width = max(ctrl_width, 12)

    for category, metric_keys in METRIC_CATEGORIES.items():
        # 데이터가 있는 메트릭만 필터
        active_keys = []
        for key in metric_keys:
            has_data = any(
                r.get('metrics', {}).get(key) is not None
                for r in results
            )
            if has_data:
                active_keys.append(key)

        if not active_keys:
            continue

        print(f'\n  ── {category} ──')

        header_parts = [f'{"Controller":<{ctrl_width}}']
        for key in active_keys:
            display_name, unit, _, _ = METRIC_DEFS[key]
            col = f'{display_name}({unit})' if unit else display_name
            header_parts.append(f'{col:>12}')
        header = ' '.join(header_parts)
        print(f'  {header}')
        print(f'  {"─" * len(header)}')

        for r in results:
            metrics = r.get('metrics', {})
            parts = [f'{r["controller"]:<{ctrl_width}}']
            for key in active_keys:
                val = metrics.get(key)
                parts.append(f'{format_metric(key, val):>12}')
            print(f'  {" ".join(parts)}')


def print_rankings(results: List[Dict]):
    """메트릭별 순위 출력"""
    rankings = compute_rankings(results, DETAIL_METRICS)

    if not rankings:
        return

    print()
    print('┌' + '─' * 60 + '┐')
    print('│  RANKINGS (Best → Worst)' + ' ' * 35 + '│')
    print('├' + '─' * 60 + '┤')

    icons = {
        'travel_time': '⏱ ',
        'position_error': '📐',
        'yaw_error': '🧭',
        'rms_jerk': '〰️',
        'mean_jerk_vx': '〰️',
        'min_obstacle_dist': '🛡 ',
        'collisions': '💥',
        'near_misses': '⚠️ ',
        'stall_ratio': '🐢',
    }

    for key, ranked in rankings.items():
        display_name, unit, fmt, _ = METRIC_DEFS[key]
        icon = icons.get(key, '  ')

        if len(ranked) < 2:
            continue

        best_ctrl, best_val = ranked[0]
        worst_ctrl, worst_val = ranked[-1]

        best_str = fmt.format(best_val)
        worst_str = fmt.format(worst_val)

        line = f'│  {icon} {display_name:<12} Best: {best_ctrl:<20} ({best_str}{unit})'
        print(f'{line:<61}│')

        if len(ranked) > 2:
            mid_ctrl, mid_val = ranked[len(ranked) // 2]
            mid_str = fmt.format(mid_val)
            mid_line = f'│     {"":12} Mid:  {mid_ctrl:<20} ({mid_str}{unit})'
            print(f'{mid_line:<61}│')

    print('└' + '─' * 60 + '┘')


def print_radar_ascii(results: List[Dict]):
    """ASCII 레이더 차트 (정규화 바 차트)"""
    if len(results) < 2:
        return

    # 정규화할 메트릭
    radar_metrics = ['travel_time', 'position_error', 'rms_jerk',
                     'min_obstacle_dist', 'mean_speed']

    print()
    print('  ── Normalized Comparison (bar = relative score, longer = better) ──')

    for key in radar_metrics:
        display_name, unit, _, lower_is_better = METRIC_DEFS.get(key, ('', '', '{}', None))
        if lower_is_better is None:
            continue

        values = []
        for r in results:
            val = r.get('metrics', {}).get(key)
            if val is not None and val != float('inf') and val != float('-inf'):
                values.append((r['controller'], float(val)))

        if len(values) < 2:
            continue

        min_val = min(v for _, v in values)
        max_val = max(v for _, v in values)
        val_range = max_val - min_val

        if val_range < 1e-9:
            continue

        print(f'\n  {display_name} ({unit}):')
        ctrl_width = max(len(c) for c, _ in values)

        for ctrl, val in values:
            if lower_is_better:
                # 낮을수록 좋으므로 반전
                score = 1.0 - (val - min_val) / val_range
            else:
                score = (val - min_val) / val_range

            bar_len = int(score * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            print(f'    {ctrl:<{ctrl_width}} {bar} {val:.3f}')


def export_csv(results: List[Dict], output_path: str):
    """결과를 CSV로 내보내기"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        header = ['controller', 'status'] + list(METRIC_DEFS.keys())
        writer.writerow(header)

        for r in results:
            row = [r.get('controller', ''), r.get('status', '')]
            metrics = r.get('metrics') or {}
            for key in METRIC_DEFS:
                row.append(metrics.get(key, ''))
            writer.writerow(row)

    print(f'  CSV exported: {output_path}')


def generate_plots(results: List[Dict], output_dir: str):
    """matplotlib 비교 플롯 생성"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print('  [WARN] matplotlib not available. Skipping plots.')
        return

    os.makedirs(output_dir, exist_ok=True)
    controllers = [r['controller'] for r in results]

    # 1. 핵심 메트릭 바 차트
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('MPPI Controller Benchmark Comparison', fontsize=14)

    plot_metrics = [
        ('travel_time', 'Travel Time (s)', True),
        ('position_error', 'Position Error (m)', True),
        ('rms_jerk', 'RMS Jerk (m/s³)', True),
        ('min_obstacle_dist', 'Min Obstacle Dist (m)', False),
        ('mean_speed', 'Mean Speed (m/s)', False),
        ('collisions', 'Collisions', True),
    ]

    for idx, (key, title, lower_better) in enumerate(plot_metrics):
        ax = axes[idx // 3][idx % 3]
        values = []
        labels = []

        for r in results:
            val = r.get('metrics', {}).get(key)
            if val is not None and val != float('inf'):
                values.append(float(val))
                labels.append(r['controller'])

        if not values:
            ax.set_title(f'{title}\n(no data)')
            continue

        colors = ['#2ecc71' if lower_better == (v == min(values))
                  or (not lower_better and v == max(values))
                  else '#3498db' for v in values]

        bars = ax.barh(range(len(values)), values, color=colors)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()

        # 값 라벨
        for bar, val in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f' {val:.2f}', va='center', fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'benchmark_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved: {plot_path}')

    # 2. 레이더 차트 (상위 6개)
    if len(results) >= 3:
        radar_keys = ['travel_time', 'position_error', 'rms_jerk',
                      'min_obstacle_dist', 'mean_speed']
        radar_labels = ['Time↓', 'PosErr↓', 'Jerk↓', 'MinObs↑', 'Speed↑']
        lower_better_flags = [True, True, True, False, False]

        # 정규화
        all_vals = {k: [] for k in radar_keys}
        for r in results:
            for k in radar_keys:
                v = r.get('metrics', {}).get(k)
                if v is not None and v != float('inf'):
                    all_vals[k].append(float(v))

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(radar_keys), endpoint=False).tolist()
        angles += angles[:1]

        for r in results[:6]:  # 상위 6개만
            scores = []
            for k, lb in zip(radar_keys, lower_better_flags):
                v = r.get('metrics', {}).get(k)
                vals = all_vals[k]
                if v is None or not vals or v == float('inf'):
                    scores.append(0.5)
                    continue
                mn, mx = min(vals), max(vals)
                rng = mx - mn
                if rng < 1e-9:
                    scores.append(0.5)
                else:
                    norm = (float(v) - mn) / rng
                    scores.append(1.0 - norm if lb else norm)

            scores += scores[:1]
            ax.plot(angles, scores, 'o-', label=r['controller'], linewidth=1.5)
            ax.fill(angles, scores, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('Controller Radar Comparison\n(outer = better)', fontsize=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

        radar_path = os.path.join(output_dir, 'benchmark_radar.png')
        plt.savefig(radar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Radar plot saved: {radar_path}')


def generate_report_from_files(filepaths: List[str], sort_by: Optional[str] = None,
                               csv_only: bool = False, plot: bool = False,
                               ranking: bool = False):
    """파일 목록에서 리포트 생성 (controller_benchmark.py --report-only에서 호출)"""
    all_results = []

    for fp in filepaths:
        data = load_results(fp)
        successful = get_successful_results(data)
        all_results.extend(successful)

        suite_name = data.get('suite_name', os.path.basename(fp))
        print(f'\n  Loaded: {fp} ({len(successful)} successful results)')

    if not all_results:
        print('  No successful results found in any file.')
        return

    if csv_only:
        csv_path = filepaths[0].replace('.json', '_report.csv')
        export_csv(all_results, csv_path)
        return

    # 요약 테이블
    print_summary_table(all_results, sort_by=sort_by)

    # 카테고리별 상세
    print_category_tables(all_results)

    # 순위
    if ranking or len(all_results) >= 3:
        print_rankings(all_results)

    # ASCII 비교
    if len(all_results) >= 2:
        print_radar_ascii(all_results)

    # 플롯
    if plot:
        output_dir = os.path.dirname(filepaths[0]) or '.'
        generate_plots(all_results, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='벤치마크 결과 리포트 생성기',
    )

    parser.add_argument('files', nargs='+', help='벤치마크 JSON 결과 파일')
    parser.add_argument('--sort-by', type=str, default=None,
                        choices=list(METRIC_DEFS.keys()),
                        help='정렬 기준 메트릭')
    parser.add_argument('--csv-only', action='store_true',
                        help='CSV만 내보내기')
    parser.add_argument('--csv-output', type=str, default=None,
                        help='CSV 출력 경로 (기본: 입력파일명_report.csv)')
    parser.add_argument('--plot', action='store_true',
                        help='matplotlib 플롯 생성')
    parser.add_argument('--ranking', action='store_true',
                        help='순위 테이블 표시')
    parser.add_argument('--detail', action='store_true',
                        help='카테고리별 상세 테이블')

    args = parser.parse_args()

    all_results = []
    for fp in args.files:
        if not os.path.exists(fp):
            print(f'  [ERROR] File not found: {fp}')
            continue

        data = load_results(fp)
        successful = get_successful_results(data)
        all_results.extend(successful)
        print(f'  Loaded: {fp} ({len(successful)} successful)')

    if not all_results:
        print('  No successful results found.')
        sys.exit(1)

    if args.csv_only:
        csv_path = args.csv_output or args.files[0].replace('.json', '_report.csv')
        export_csv(all_results, csv_path)
        return

    # 요약
    print_summary_table(all_results, sort_by=args.sort_by)

    # 상세
    if args.detail:
        print_category_tables(all_results)

    # 순위
    if args.ranking or len(all_results) >= 3:
        print_rankings(all_results)

    # ASCII 비교
    if len(all_results) >= 2:
        print_radar_ascii(all_results)

    # 플롯
    if args.plot:
        output_dir = os.path.dirname(args.files[0]) or '.'
        generate_plots(all_results, output_dir)

    # CSV도 항상 내보내기
    csv_path = args.csv_output or args.files[0].replace('.json', '_report.csv')
    export_csv(all_results, csv_path)


if __name__ == '__main__':
    main()
