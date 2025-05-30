import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

# ---------------------------- 微分方程定义 ----------------------------
def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组实现
    
    参数:
        state: 包含位置x和速度v的数组 [x, v]
        t: 时间（未使用）
        omega: 角频率
    
    返回:
        导数数组 [dx/dt, dv/dt]
    """
    x, v = state
    dxdt = v  # 位置变化率 = 速度
    dvdt = -omega**2 * x  # 速度变化率 = -ω²x
    return np.array([dxdt, dvdt])

def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组实现
    
    参数:
        state: 包含位置x和速度v的数组 [x, v]
        t: 时间（未使用）
        omega: 角频率
    
    返回:
        导数数组 [dx/dt, dv/dt]
    """
    x, v = state
    dxdt = v  # 位置变化率 = 速度
    dvdt = -omega**2 * x**3  # 速度变化率 = -ω²x³
    return np.array([dxdt, dvdt])

# ---------------------------- 数值积分方法 ----------------------------
def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    四阶龙格-库塔方法（RK4）单步积分
    
    参数:
        ode_func: 微分方程函数
        state: 当前状态向量
        t: 当前时间
        dt: 时间步长
        **kwargs: 传递给ode_func的参数
    
    返回:
        下一时刻的状态向量
    """
    k1 = ode_func(state, t, **kwargs) * dt
    k2 = ode_func(state + 0.5*k1, t + 0.5*dt, **kwargs) * dt
    k3 = ode_func(state + 0.5*k2, t + 0.5*dt, **kwargs) * dt
    k4 = ode_func(state + k3, t + dt, **kwargs) * dt
    
    new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
    return new_state

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组
    
    参数:
        ode_func: 微分方程函数
        initial_state: 初始状态向量
        t_span: 时间范围 (t_start, t_end)
        dt: 时间步长
        **kwargs: 传递给ode_func的参数
    
    返回:
        t_values: 时间点数组
        states: 状态矩阵（每行对应一个时间点的状态）
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt) + 1
    t_values = np.linspace(t_start, t_end, num_steps)
    states = np.zeros((num_steps, len(initial_state)))
    states[0] = initial_state
    
    for i in range(1, num_steps):
        states[i] = rk4_step(ode_func, states[i-1], t_values[i-1], dt, **kwargs)
    
    return t_values, states

# ---------------------------- 绘图函数 ----------------------------
def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制位移随时间的变化曲线
    
    参数:
        t: 时间点数组
        states: 状态矩阵（每行包含x和v）
        title: 图标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Displacement x(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹（速度 vs 位移）
    
    参数:
        states: 状态矩阵（每行包含x和v）
        title: 图标题
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], label='Phase Space Trajectory')
    plt.xlabel('Displacement (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# ---------------------------- 周期分析 ----------------------------
def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    通过过零点检测计算振动周期
    
    参数:
        t: 时间点数组
        states: 状态矩阵（每行包含x和v）
    
    返回:
        平均周期（单位：秒）
    """
    x = states[:, 0]
    # 寻找过零点（从正到负）
    zero_crossings = np.where(np.diff(np.sign(x)) < 0)[0]
    
    if len(zero_crossings) < 2:
        return np.nan  # 无法计算周期
    
    # 计算相邻过零点的时间差
    periods = np.diff(t[zero_crossings])
    return np.mean(periods)  # 修复：相邻过零点间隔即为完整周期，无需乘以2

# ---------------------------- 主程序 ----------------------------
def main():
    # 参数设置
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    
    # ========== 任务1：简谐振子求解 ==========
    initial_state = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
    t_harmonic, states_harmonic = solve_ode(
        harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega
    )
    plot_time_evolution(t_harmonic, states_harmonic, 'Harmonic Oscillator: Time Evolution')
    plot_phase_space(states_harmonic, 'Harmonic Oscillator: Phase Space')
    
    # ========== 任务2：振幅对周期的影响 ==========
    amplitudes = [1.0, 2.0, 3.0]
    print("简谐振子周期分析：")
    for amp in amplitudes:
        t, states = solve_ode(
            harmonic_oscillator_ode, 
            np.array([amp, 0.0]), 
            t_span, 
            dt, 
            omega=omega
        )
        period = analyze_period(t, states)
        print(f"振幅 {amp} m -> 周期 {period:.3f} s")
    
    # ========== 任务3：非谐振子分析 ==========
    initial_state_an = np.array([1.0, 0.0])
    t_anharmonic, states_anharmonic = solve_ode(
        anharmonic_oscillator_ode, initial_state_an, t_span, dt, omega=omega
    )
    plot_time_evolution(t_anharmonic, states_anharmonic, 'Anharmonic Oscillator: Time Evolution')
    
    # 非谐振子周期分析
    print("\n非谐振子周期分析：")
    for amp in amplitudes:
        t, states = solve_ode(
            anharmonic_oscillator_ode, 
            np.array([amp, 0.0]), 
            t_span, 
            dt, 
            omega=omega
        )
        period = analyze_period(t, states)
        print(f"振幅 {amp} m -> 周期 {period:.3f} s")
    
    # ========== 任务4：相空间分析 ==========
    plot_phase_space(states_anharmonic, 'Anharmonic Oscillator: Phase Space')

if __name__ == "__main__":
    main()
