
**Test версия**

from neurosdk.scanner import Scanner
from neurosdk.sensor import Sensor
from neurosdk.brainbit_sensor import BrainBitSensor
from neurosdk.cmn_types import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
from matplotlib.gridspec import GridSpec
import time


SAMPLE_RATE = 250  # Частота дискретизации
TIME_WINDOW = 10   # Временное окно в секундах
BUFFER_SIZE = int(SAMPLE_RATE * TIME_WINDOW)  # Размер буфера

# буферы для каждого канала
buffers = {
    'O1': deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE),
    'O2': deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE),
    'T3': deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE),
    'T4': deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
}

# врем шкала
time_axis = np.linspace(-TIME_WINDOW, 0, BUFFER_SIZE, endpoint=False)

#графики
fig = plt.figure(figsize=(15, 10))
fig.suptitle('BrainBit EEG Data - Live Streaming', fontsize=16, fontweight='bold')

# расположение
gs = GridSpec(4, 2, figure=fig, height_ratios=[3, 3, 3, 1], hspace=0.4, wspace=0.3)

# графики на каждый канал
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Один общий график
ax5 = fig.add_subplot(gs[2, :])

# Текстовая инфа
ax6 = fig.add_subplot(gs[3, :])
ax6.axis('off')

# Цвета
colors = {
    'O1': '#FF6B6B',
    'O2': '#4ECDC4',
    'T3': '#45B7D1',
    'T4': '#96CEB4'
}

# линия в реальном времени для графиков
lines = {}
for ax, channel in zip([ax1, ax2, ax3, ax4], ['O1', 'O2', 'T3', 'T4']):
    lines[channel], = ax.plot(time_axis, buffers[channel],
                              color=colors[channel], linewidth=1.5, alpha=0.8)
    ax.set_title(f'Channel {channel}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-TIME_WINDOW, 0)
    ax.set_ylim(-200, 200)  # Начальные пределы по Y

# Линии для общего
common_lines = {}
for channel in ['O1', 'O2', 'T3', 'T4']:
    common_lines[channel], = ax5.plot(time_axis, buffers[channel],
                                      color=colors[channel], linewidth=1,
                                      alpha=0.7, label=f'Channel {channel}')
ax5.set_title('All Channels Combined', fontsize=12, fontweight='bold')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Amplitude (µV)')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(-TIME_WINDOW, 0)
ax5.set_ylim(-200, 200)
ax5.legend(loc='upper right', fontsize=9)

# Текстовая инфа
info_text = ax6.text(0.02, 0.5, '', fontsize=10, fontfamily='monospace',
                     verticalalignment='center', horizontalalignment='left')

# подключение к устройству
def connect_to_device():
    """Подключение к BrainBit устройству"""
    try:
        print("Scanning for BrainBit devices...")
        scanner = Scanner([SensorFamily.LEBrainBit])
        scanner.start()
        time.sleep(55)

        sensors = scanner.sensors()
        if not sensors:
            print("No BrainBit devices found!")
            return None

        print(f"Found device: {sensors[0].Name} at {sensors[0].Address}")
        sensor = scanner.create_sensor(sensors[0])

        # обработчик
        sensor.signalDataReceived = on_signal_data_received

        print("Device connected successfully!")
        return sensor, scanner

    except Exception as e:
        print(f"Error connecting to device: {e}")
        return None

# работа с данными
def on_signal_data_received(sensor, data):
    """Обработчик принимаемых данных"""
    for data_point in data:
        # буферы для данных
        buffers['O1'].append(data_point.O1 * 1000000)  # в микровольтах?
        buffers['O2'].append(data_point.O2 * 1000000)
        buffers['T3'].append(data_point.T3 * 1000000)
        buffers['T4'].append(data_point.T4 * 1000000)

# обновление графиков
def update_plot(frame):
    """Обновление графиков"""
    # линии на графиках
    for channel in ['O1', 'O2', 'T3', 'T4']:
        lines[channel].set_ydata(buffers[channel])
        common_lines[channel].set_ydata(buffers[channel])

    # масштабирование
    for ax, channel in zip([ax1, ax2, ax3, ax4], ['O1', 'O2', 'T3', 'T4']):
        current_data = list(buffers[channel])
        if current_data:
            y_min = min(current_data)
            y_max = max(current_data)
            margin = max(abs(y_min), abs(y_max)) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)

    # масштабирование общего графика
    all_data = []
    for channel in ['O1', 'O2', 'T3', 'T4']:
        all_data.extend(list(buffers[channel]))

    if all_data:
        y_min = min(all_data)
        y_max = max(all_data)
        margin = max(abs(y_min), abs(y_max)) * 0.1
        ax5.set_ylim(y_min - margin, y_max + margin)

    # обновление инфы
    update_info_text()

    return [lines[channel] for channel in ['O1', 'O2', 'T3', 'T4']] + \
           [common_lines[channel] for channel in ['O1', 'O2', 'T3', 'T4']] + \
           [info_text]

def update_info_text():
    current_time = time.strftime('%H:%M:%S')

    # статистика
    stats = {}
    for channel in ['O1', 'O2', 'T3', 'T4']:
        data = list(buffers[channel])
        if data:
            stats[channel] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': min(data),
                'max': max(data)
            }
        else:
            stats[channel] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

    # текст
    info_str = f"Time: {current_time} | "
    info_str += f"Buffer: {len(list(buffers['O1']))}/{BUFFER_SIZE} samples | "
    info_str += f"Window: {TIME_WINDOW}s\n"
    info_str += "─" * 80 + "\n"

    for channel in ['O1', 'O2', 'T3', 'T4']:
        info_str += (f"{channel}: Mean={stats[channel]['mean']:7.2f}µV | "
                     f"Std={stats[channel]['std']:6.2f}µV | "
                     f"Range=[{stats[channel]['min']:6.1f}, {stats[channel]['max']:6.1f}]µV\n")

    info_text.set_text(info_str)


def main():
    """Основная функция"""
    print("=====================================")
    print("BrainBit EEG Real-time Visualization")
    print("=====================================")

    # конект
    result = connect_to_device()
    if not result:
        print("Failed to connect")
        return

    sensor, scanner = result

    try:
        # анимация обновления
        ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True, cache_frame_data=False)

        #
        def on_close(event):
            print("\nClosing...")
            if sensor:
                sensor.signalDataReceived = None
            if scanner:
                scanner.stop()
                scanner.dispose()
            plt.close('all')
            print("Disconnected")

        fig.canvas.mpl_connect('close_event', on_close)

        #  графики
        plt.tight_layout()
        plt.show()

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Очистка
        if sensor:
            sensor.signalDataReceived = None
        if scanner:
            scanner.stop()
            scanner.dispose()
        print("Application closed.")


if __name__ == "__main__":
    main()
