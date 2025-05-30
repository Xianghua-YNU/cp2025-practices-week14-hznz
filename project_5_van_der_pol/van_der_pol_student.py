import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """
    van der Pol振子的一阶微分方程组。
    """
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dxdt, dvdt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    """
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = ode_func(state + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = ode_func(state + dt * k3, t + dt, **kwargs)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt) + 1
    t_array = np.linspace(t_start, t_end, num_steps)
    states = np.zeros((num_steps, len(initial_state)))
    states[0] = initial_state

    for i in range(1, num_steps):
        states[i] = rk4_step(ode_func, states[i-1], t_array[i-1], dt, **kwargs)
    
    return t_array, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t, states[:, 0], label='位移 x(t)')
    plt.plot(t, states[:, 1], label='速度 v(t)')
    plt.xlabel('时间 t')
    plt.ylabel('状态')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], label='相空间轨迹')
    plt.xlabel('位移 x')
    plt.ylabel('速度 v')
    plt.title(title)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.show()

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """
    计算van der Pol振子的能量。
    """
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

def analyze_limit_cycle(states: np.ndarray, dt: float) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）。
    """
    # 计算振幅
    amplitude = np.max(np.abs(states[:, 0]))
    
    # 改进周期计算
    x_values = states[:, 0]
    # 找到所有从负到正的过零点
    zero_crossings = np.where((x_values[:-1] <= 0) & (x_values[1:] > 0))[0]
    
    if len(zero_crossings) >= 2:
        # 计算相邻过零点之间的时间差
        crossing_times = zero_crossings * dt
        periods = np.diff(crossing_times)
        period = np.mean(periods)
    else:
        period = np.nan
    
    return amplitude, period

def plot_energy_evolution(t: np.ndarray, states: np.ndarray, omega: float, title: str) -> None:
    """
    绘制能量随时间的变化。
    """
    energy = np.array([calculate_energy(state, omega) for state in states])
    plt.figure(figsize=(10, 5))
    plt.plot(t, energy)
    plt.xlabel('时间 t')
    plt.ylabel('能量 E')
    plt.title(title)
    plt.grid(True)
    plt.show()

def main():
    # 设置基本参数
    omega = 1.0
    t_span = (0, 20)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # 任务1 - 基本实现
    print("=== 任务1: 基本实现 ===")
    mu = 1.0
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f"van der Pol振子时间演化 (μ={mu})")
    plot_phase_space(states, f"van der Pol振子相空间轨迹 (μ={mu})")
    plot_energy_evolution(t, states, omega, f"van der Pol振子能量变化 (μ={mu})")
    
    amplitude, period = analyze_limit_cycle(states, dt)
    print(f"μ={mu}时的极限环特征:")
    print(f"振幅: {amplitude:.4f}")
    print(f"周期: {period:.4f}")
    
    # 任务2 - 参数影响分析
    print("\n=== 任务2: 参数影响分析 ===")
    mu_values = [1.0, 2.0, 4.0]
    
    plt.figure(figsize=(10, 5))
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plt.plot(t, states[:, 0], label=f'μ={mu}')
    plt.xlabel('时间 t')
    plt.ylabel('位移 x(t)')
    plt.title('不同μ值下的van der Pol振子位移比较')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 8))
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plt.plot(states[:, 0], states[:, 1], label=f'μ={mu}')
    plt.xlabel('位移 x')
    plt.ylabel('速度 v')
    plt.title('不同μ值下的van der Pol振子相空间轨迹')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()
    
    # 任务3 - 初始条件影响分析
    print("\n=== 任务3: 初始条件影响分析 ===")
    mu = 2.0
    initial_conditions = [
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([0.5, 1.0]),
        np.array([0.0, 2.0])
    ]
    
    plt.figure(figsize=(8, 8))
    for i, ic in enumerate(initial_conditions):
        t, states = solve_ode(van_der_pol_ode, ic, t_span, dt, mu=mu, omega=omega)
        plt.plot(states[:, 0], states[:, 1], label=f'初始条件 {i+1}')
    plt.xlabel('位移 x')
    plt.ylabel('速度 v')
    plt.title(f'不同初始条件下的van der Pol振子相空间轨迹 (μ={mu})')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    main()
