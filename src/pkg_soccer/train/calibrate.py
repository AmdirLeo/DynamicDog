"""从 `record_motion.py` 的 CSV 拟合实机的响应特性，用来替换仿真里的猜测值。

仿真的三条执行假设目前都是猜的：指令延迟 `latency=(1,3)` 帧、一阶惯性
`tau=(0.08,0.25)` s、转向惯性 `tau_w=(0.10,0.30)` s（见 `EnvConfig`）。
本工具把"指令 → 动捕实测机身速度"拟合成 `延迟 + 一阶惯性 + 增益`，
直接给出应该写进训练环境的参数。

CSV 由包里自带的记录器产生（只读、不发指令）：

    ros2 run pkg_soccer soccer --ros-args -p enable_motion:=true ...
    ros2 run pkg_soccer record_motion --ros-args -p output:=/abs/path/id.csv

必用列：`time_s, cmd_vx_mps, cmd_vy_mps, cmd_wz_radps, cmd_type,
        dog_vx_body_mps, dog_vy_body_mps, dog_wz_radps, dog_valid`

用法：
    python -m train.calibrate /abs/path/id.csv
    python -m train.calibrate id.csv --grid-tau 0.02,0.8 --max-delay 10

模型（与 `team_env.py` 的指令通道一致）：

    v[k+1] = v[k] + dt/tau * (K * cmd[k-d] - v[k])

`d` 用整数帧、`tau`/`K` 在给定延迟下用最小二乘解出，最后按 RMSE 选最优。
注意：**台阶越陡越准**（直走→停、停→直走），正弦激励能同时估出相位延迟。
"""

import argparse
import csv
import math

import numpy as np

COLUMNS = ('time_s', 'cmd_vx_mps', 'cmd_vy_mps', 'cmd_wz_radps',
           'dog_vx_body_mps', 'dog_vy_body_mps', 'dog_wz_radps')


def load(path):
    """读成等间隔 20 Hz 序列，返回 dict（缺列直接报错，避免静默算出假参数）。"""
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f'{path} 缺少列 {missing}；请用 record_motion.py 重新录')
        rows = [row for row in reader]
    if len(rows) < 100:
        raise SystemExit(f'只有 {len(rows)} 行，至少需要 100 行（5 s @ 20 Hz）')
    time = np.array([float(r['time_s']) for r in rows])
    data = {}
    for name in COLUMNS[1:]:
        data[name] = np.array([float(r[name] or 'nan') for r in rows])
    # 只在狗定位有效、且不是结束帧时拟合
    valid = np.ones(len(rows), bool)
    if 'dog_valid' in rows[0]:
        valid &= np.array([str(r['dog_valid']).strip().lower()
                           not in ('0', 'false', '', 'none') for r in rows])
    if 'cmd_type' in rows[0]:
        valid &= np.array([str(r['cmd_type']).strip().upper() not in ('2', 'END', 'SERVO_END')
                           for r in rows])
    valid &= np.isfinite(data['dog_vx_body_mps']) & np.isfinite(data['cmd_vx_mps'])
    grid = np.arange(time[0], time[-1], 0.05)
    series = {'t': grid}
    for name in COLUMNS[1:]:
        values = np.where(valid, data[name], np.nan)
        filled = np.interp(grid, time, np.nan_to_num(values))
        series[name] = filled
    series['valid'] = np.interp(grid, time, valid.astype(float)) > 0.999
    return series


def fit_first_order(cmd, measured, mask, delay, tau):
    """固定延迟与 tau 时，最小二乘解增益 K 并算 RMSE。"""
    dt = 0.05
    v = np.zeros(len(cmd))
    v[0] = measured[mask][0] if mask.any() else 0.0
    shifted = np.zeros(len(cmd))
    if delay > 0:
        shifted[delay:] = cmd[:-delay]
    else:
        shifted = cmd.copy()
    alpha = dt / tau
    for k in range(len(cmd) - 1):
        v[k + 1] = v[k] + alpha * (shifted[k] - v[k])
    # 只在有指令的样本上解 K：measured ≈ K * v（v 是对单位增益的响应）
    use = mask & (np.abs(v) > 1e-6)
    if use.sum() < 20:
        return 1.0, float('inf')
    K = float(np.dot(measured[use], v[use]) / np.dot(v[use], v[use]))
    K = float(np.clip(K, 0.2, 2.0))
    residual = measured[mask] - K * v[mask]
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return K, rmse


def fit_axis(series, cmd_name, meas_name, max_delay=10, tau_range=(0.02, 0.8)):
    cmd = series[cmd_name]
    measured = series[meas_name]
    mask = (series['valid'] & np.isfinite(measured) & np.isfinite(cmd) &
            (np.abs(cmd) > 1e-3))
    if mask.sum() < 50:
        return None
    best = None
    for delay in range(0, max_delay + 1):
        for tau in np.arange(tau_range[0], tau_range[1] + 1e-9, 0.01):
            K, rmse = fit_first_order(cmd, measured, mask, delay, tau)
            if best is None or rmse < best[2]:
                best = (delay, float(tau), rmse, K)
    delay, tau, rmse, K = best
    scale = float(np.std(measured[mask])) or 1.0
    return {'delay': delay, 'tau': tau, 'K': K, 'rmse': rmse,
            'nr2': 1.0 - (rmse / scale) ** 2, 'n': int(mask.sum())}


def self_test(path='_selftest_motion.csv', delay=2, tau=0.15, gain=0.95,
              noise=0.01, seconds=40.0, seed=0):
    """生成一份"已知答案"的假 CSV，验证拟合能把这些数捞回来。"""
    rng = np.random.default_rng(seed)
    dt = 0.05
    steps = int(seconds / dt)
    t = np.arange(steps) * dt
    # 随机分段常值 + 正弦转向：既有台阶（估 τ）也有相位信息（估延迟）
    cmd_vx = np.repeat(rng.uniform(0.1, 1.2, steps // 20 + 1), 20)[:steps]
    cmd_vy = np.zeros(steps)
    cmd_wz = 0.5 * np.sin(2 * math.pi * 0.2 * t)
    vx = np.zeros(steps)
    vy = np.zeros(steps)
    wz = np.zeros(steps)
    d = delay
    for k in range(steps - 1):
        cx = gain * cmd_vx[k - d] if k >= d else 0.0
        vx[k + 1] = vx[k] + dt / tau * (cx - vx[k])
        wz[k + 1] = wz[k] + dt / 0.2 * ((0.5 * cmd_wz[k - d] if k >= d else 0.0)
                                        - wz[k])
    vx += rng.normal(0, noise, steps)
    wz += rng.normal(0, noise * 0.5, steps)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(list(COLUMNS) + ['cmd_type', 'dog_valid'])
        for k in range(steps):
            writer.writerow([f'{t[k]:.3f}', f'{cmd_vx[k]:.4f}', f'{cmd_vy[k]:.4f}',
                             f'{cmd_wz[k]:.4f}', f'{vx[k]:.4f}', f'{vy[k]:.4f}',
                             f'{wz[k]:.4f}', '1', '1'])
    print(f'已生成 {path}（真值：延迟 {delay} 帧、τ={tau:.2f} s、K={gain:.2f}、'
          f'噪声 {noise} m/s）')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description='从 record_motion CSV 拟合执行特性')
    parser.add_argument('csv', nargs='?', default='',
                        help='record_motion.py 输出的 CSV')
    parser.add_argument('--self-test', action='store_true',
                        help='用合成数据验证拟合器本身（不需要实机数据）')
    parser.add_argument('--max-delay', type=int, default=10,
                        help='最大指令延迟（帧，20 Hz）')
    parser.add_argument('--grid-tau', default='0.02,0.80')
    args = parser.parse_args(argv)

    low, high = (float(v) for v in args.grid_tau.split(','))
    if args.self_test:
        args.csv = self_test()
    if not args.csv:
        raise SystemExit('请给出 CSV 路径，或加 --self-test')
    series = load(args.csv)
    span = series['t'][-1] - series['t'][0]
    print(f'样本 {len(series["t"])} 帧 / {span:.1f} s，'
          f'有效 {int(series["valid"].sum())} 帧')

    rows = {}
    for label, cmd_name, meas_name in (
            ('前进 vx', 'cmd_vx_mps', 'dog_vx_body_mps'),
            ('侧移 vy', 'cmd_vy_mps', 'dog_vy_body_mps'),
            ('转向 wz', 'cmd_wz_radps', 'dog_wz_radps')):
        result = fit_axis(series, cmd_name, meas_name, args.max_delay,
                          (low, high))
        rows[label] = result
        if result is None:
            print(f'{label}: 样本不足，跳过')
            continue
        print(f'{label}: 延迟 {result["delay"]} 帧'
              f'（{result["delay"] * 0.05 * 1000:.0f} ms）、'
              f'τ={result["tau"]:.2f} s、增益 K={result["K"]:.2f}、'
              f'RMSE={result["rmse"]:.3f} m/s（1-R²/σ={result["nr2"]:.2f}）')

    linear = [rows.get('前进 vx'), rows.get('侧移 vy')]
    if all(linear):
        delays = [r['delay'] for r in linear]
        taus = [r['tau'] for r in linear]
        print('\n建议写进 train/team_env.py 的 EnvConfig（各放宽 ~30% 做随机化）：')
        print(f'  latency=({max(0, min(delays))}, {max(delays) + 1})')
        print(f'  tau=({max(0.04, min(taus) * 0.7):.2f}, '
              f'{max(taus) * 1.3:.2f})')
        if rows.get('转向 wz'):
            tw = rows['转向 wz']['tau']
            print(f'  tau_w=({max(0.05, tw * 0.7):.2f}, {tw * 1.3:.2f})')
        print('  若是明显偏低（<0.6），说明指令到不了位：先查固件限幅/打滑，'
              '再考虑把速度档降下来')
    print('\n做完这步再重训，才算"动力学标定过"；否则仿真里的延迟/惯性只是猜的。')


if __name__ == '__main__':
    main()
