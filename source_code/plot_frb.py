import h5py
import matplotlib.pyplot as plt
import os
import glob

frb_dir = "FRB"
h5_files = glob.glob(os.path.join(frb_dir, "*.h5"))

if not h5_files:
    print("Ошибка! Не найдено файлов .h5 в папке FRB")
    exit()

# Берем первый найденный файл для визуализации
h5_path = h5_files[0]
print(f"[+] Строю график для файла: {os.path.basename(h5_path)}")

with h5py.File(h5_path, "r") as f:
    group = f['frb']
    
    # Извлекаем все необходимые массивы, которые мы разобрали
    wfall = group['wfall'][:]
    ts = group['ts'][:]
    spec = group['spec'][:]
    plot_time = group['plot_time'][:]
    plot_freq = group['plot_freq'][:]
    extent = group['extent'][:]

# Создаем сетку для красивого расположения графиков
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 3],
                      wspace=0.1, hspace=0.1)

# 1. Верхний график: Временной профиль (Time Series)
ax_ts = fig.add_subplot(gs[0, 1], sharex=None)
ax_ts.plot(plot_time, ts, color='crimson', lw=2)
ax_ts.set_title(f"Радиовсплеск: {os.path.basename(h5_path)}", fontsize=14, pad=15)
ax_ts.set_ylabel("Интенсивность")
ax_ts.grid(True, linestyle='--', alpha=0.5)
ax_ts.xaxis.set_tick_params(labelbottom=False) # убираем нижние метки

# 2. Левый график: Спектр (Spectrum)
ax_spec = fig.add_subplot(gs[1, 0])
ax_spec.plot(spec, plot_freq, color='royalblue', lw=2)
ax_spec.set_xlabel("Интенсивность")
ax_spec.set_ylabel("Частота (МГц)")
ax_spec.invert_xaxis() # переворачиваем, чтобы график примыкал к водопаду
ax_spec.grid(True, linestyle='--', alpha=0.5)

# 3. Центральный график: Водопад (Dynamic Spectrum)
ax_wfall = fig.add_subplot(gs[1, 1], sharex=ax_ts, sharey=ax_spec)
# extent определяет физические границы осей [время_мин, время_макс, частота_мин, частота_макс]
im = ax_wfall.imshow(wfall, aspect='auto', cmap='viridis', extent=extent, origin='lower')
ax_wfall.set_xlabel("Время (мс)")
ax_wfall.yaxis.set_tick_params(labelleft=False) # убираем левые метки, они есть на спектре

# Добавляем цветовую шкалу справа
cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.5])
fig.colorbar(im, cax=cbar_ax, label='Яркостная температура / Поток')

# Сохраняем результат в картинку
output_image = h5_path.replace(".h5", "_visualization.png")
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"[ ✔️ ] График успешно сохранен как: {output_image}")

# Показываем на экране (если запускаете в GUI среде)
plt.show()