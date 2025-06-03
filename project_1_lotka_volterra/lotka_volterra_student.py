#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 - 学生代码模板

学生姓名：[胡正一]
学号：[20231050047]
完成日期：[5.28]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================== 核心函数实现 ==================

def lotka_volterra_system(state: np.ndarray, t: float, alpha: float, beta: float, 
                          gamma: float, delta: float) -> np.ndarray:
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = gamma * x * y - delta * y
    return np.array([dxdt, dydt])


def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        y_current = y[i]
        t_current = t[i]
        derivative = f(y_current, t_current, *args)
        y_next = y_current + dt * derivative
        y[i+1] = y_next
    
    return t, y


def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                         dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        current_y = y[i]
        current_t = t[i]
        k1 = dt * f(current_y, current_t, *args)
        k2 = dt * f(current_y + k1, current_t + dt, *args)
        y_next = current_y + (k1 + k2) / 2
        y[i+1] = y_next
    
    return t, y


def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float], 
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        current_y = y[i]
        current_t = t[i]
        k1 = dt * f(current_y, current_t, *args)
        k2 = dt * f(current_y + k1/2, current_t + dt/2, *args)
        k3 = dt * f(current_y + k2/2, current_t + dt/2, *args)
        k4 = dt * f(current_y + k3, current_t + dt, *args)
        y_next = current_y + (k1 + 2*k2 + 2*k3 + k4) / 6
        y[i+1] = y_next
    
    return t, y


def solve_lotka_volterra(alpha: float, beta: float, gamma: float, delta: float,
                        x0: float, y0: float, t_span: Tuple[float, float], 
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0_vec = np.array([x0, y0])
    t, y = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x = y[:, 0]
    y_values = y[:, 1]
    return t, x, y_values


def compare_methods(alpha: float, beta: float, gamma: float, delta: float,
                   x0: float, y0: float, t_span: Tuple[float, float], 
                   dt: float) -> Dict[str, Dict]:
    y0_vec = np.array([x0, y0])
    args = (alpha, beta, gamma, delta)
    
    # 三种方法求解
    t_euler, y_euler = euler_method(lotka_volterra_system, y0_vec, t_span, dt, *args)
    t_ie, y_ie = improved_euler_method(lotka_volterra_system, y0_vec, t_span, dt, *args)
    t_rk4, y_rk4 = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, *args)
    
    # 结果整理
    results = {
        'euler': {'t': t_euler, 'x': y_euler[:,0], 'y': y_euler[:,1]},
        'improved_euler': {'t': t_ie, 'x': y_ie[:,0], 'y': y_ie[:,1]},
        'rk4': {'t': t_rk4, 'x': y_rk4[:,0], 'y': y_rk4[:,1]}
    }
    return results


# ================== 可视化函数 ==================

def plot_population_dynamics(t: np.ndarray, x: np.ndarray, y: np.ndarray, 
                           title: str = "Lotka-Volterra种群动力学") -> None:
    plt.figure(figsize=(12, 5))
    
    # 时间序列图
    plt.subplot(1, 2, 1)
    plt.plot(t, x, label='猎物 (x)', color='green')
    plt.plot(t, y, label='捕食者 (y)', color='red')
    plt.xlabel('时间')
    plt.ylabel('种群数量')
    plt.title(f'{title}\n时间序列')
    plt.grid(True)
    plt.legend()
    
    # 相空间轨迹图
    plt.subplot(1, 2, 2)
    plt.plot(x, y, color='blue')
    plt.xlabel('猎物数量 (x)')
    plt.ylabel('捕食者数量 (y)')
    plt.title(f'{title}\n相空间轨迹')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_method_comparison(results: Dict) -> None:
    methods = ['euler', 'improved_euler', 'rk4']
    titles = ['欧拉法', '改进欧拉法', '4阶龙格-库塔法']
    
    plt.figure(figsize=(15, 8))
    
    # 绘制时间序列比较
    for i, method in enumerate(methods):
        plt.subplot(2, 3, i+1)
        plt.plot(results[method]['t'], results[method]['x'], label='猎物', color='green')
        plt.plot(results[method]['t'], results[method]['y'], label='捕食者', color='red')
        plt.title(f'{titles[i]} 时间序列')
        plt.xlabel('时间')
        plt.ylabel('种群数量')
        plt.grid(True)
        plt.legend()
    
    # 绘制相空间比较
    for i, method in enumerate(methods):
        plt.subplot(2, 3, i+4)
        plt.plot(results[method]['x'], results[method]['y'], color='blue')
        plt.title(f'{titles[i]} 相空间')
        plt.xlabel('猎物数量')
        plt.ylabel('捕食者数量')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# ================== 参数分析函数 ==================

def analyze_parameters() -> None:
    # 基础参数
    base_params = (1.0, 0.5, 0.5, 2.0)
    t_span = (0, 30)
    dt = 0.01
    
    # 不同初始条件测试
    initial_conditions = [(2, 2), (1, 1), (3, 1), (1, 3)]
    plt.figure(figsize=(12, 6))
    for i, (x0, y0) in enumerate(initial_conditions):
        t, x, y = solve_lotka_volterra(*base_params, x0, y0, t_span, dt)
        plt.subplot(2, 2, i+1)
        plt.plot(t, x, label=f'x0={x0}, y0={y0}')
        plt.plot(t, y)
        plt.title(f'初始条件 ({x0}, {y0})')
        plt.xlabel('时间')
        plt.ylabel('种群数量')
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 守恒量验证
    t, x, y = solve_lotka_volterra(*base_params, 2, 2, t_span, dt)
    alpha, beta, gamma, delta = base_params
    H = gamma * x + beta * y - delta * np.log(x) - alpha * np.log(y)
    plt.figure()
    plt.plot(t, H)
    plt.title('守恒量 H 随时间变化')
    plt.xlabel('时间')
    plt.ylabel('H 值')
    plt.grid(True)
    plt.show()


# ================== 主函数 ==================

def main():
    # 参数设置
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t_span = (0, 30)
    dt = 0.01
    
    print("=== Lotka-Volterra捕食者-猎物模型分析 ===")
    print(f"参数: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    print(f"初始条件: x0={x0}, y0={y0}")
    print(f"时间范围: {t_span}, 步长: {dt}")
    
    try:
        # 1. 基本求解与绘图
        print("\n1. 使用4阶龙格-库塔法求解并绘图...")
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plot_population_dynamics(t, x, y)
        
        # 2. 方法比较
        print("\n2. 比较不同数值方法...")
        results = compare_methods(alpha, beta, gamma, delta, x0, y0, (0, 10), 0.1)
        plot_method_comparison(results)
        
        # 3. 参数分析
        print("\n3. 分析参数影响...")
        analyze_parameters()
        
        # 4. 数值结果统计
        print("\n4. 数值结果统计:")
        print(f"猎物数量范围: [{np.min(x):.2f}, {np.max(x):.2f}]")
        print(f"捕食者数量范围: [{np.min(y):.2f}, {np.max(y):.2f}]")
        from scipy.signal import find_peaks
        peaks_x, _ = find_peaks(x, height=np.mean(x))
        if len(peaks_x) > 1:
            period = np.mean(np.diff(t[peaks_x]))
            print(f"估计周期: {period:.3f}")
        
    except Exception as e:
        print(f"\n运行时发生错误: {e}")


if __name__ == "__main__":
    main()
