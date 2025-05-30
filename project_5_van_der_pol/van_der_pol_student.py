import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, List

import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

# 然后继续您的绘图代码

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """
    van der Pol振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        mu: float, 非线性阻尼参数
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dxdt, dvdt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
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
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    # TODO: 实现ODE求解器
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
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现时间演化图的绘制
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
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现相空间图的绘制
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('位移 x')
    plt.ylabel('速度 v')
    plt.title(title)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()
    
def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """
    计算van der Pol振子的能量。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        omega: float, 角频率
    
    返回:
        float: 系统的能量
    """
    # TODO: 实现能量计算
    # E = (1/2)v^2 + (1/2)omega^2*x^2
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）。
    
    参数:
        states: np.ndarray, 状态数组
    
    返回:
        Tuple[float, float]: (振幅, 周期)
    """
    # TODO: 实现极限环分析
    # 计算振幅为x的最大绝对值
    amplitude = np.max(np.abs(states[:, 0]))
    
    # 计算周期 - 简单估计为过零点的平均间隔
    x_values = states[:, 0]
    zero_crossings = np.where(np.diff(np.sign(x_values)))[0]
    if len(zero_crossings) > 1:
        periods = np.diff(zero_crossings)
        period = np.mean(periods)
    else:
        period = np.nan
    
    return amplitude, period

def main():
    # Set basic parameters
    mu = 1.0
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # Task 1 - Basic implementation
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f'Time Evolution of van der Pol Oscillator (μ={mu})')
    
    # Task 2 - Parameter influence analysis
    mu_values = [1.0, 2.0, 4.0]
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_time_evolution(t, states, f'Time Evolution of van der Pol Oscillator (μ={mu})')
        amplitude, period = analyze_limit_cycle(states)
        print(f'μ = {mu}: Amplitude ≈ {amplitude:.3f}, Period ≈ {period*dt:.3f}')
    
    # Task 3 - Phase space analysis
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, f'Phase Space Trajectory of van der Pol Oscillator (μ={mu})')

if __name__ == "__main__":
    main()
