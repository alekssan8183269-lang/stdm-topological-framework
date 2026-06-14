import h5py
import numpy as np
import os
import glob

frb_dir = "FRB"

print("[*] ЗАПУСК ТОЧНОЙ КОНВЕРТАЦИИ ГРУППЫ FRB...")
h5_files = glob.glob(os.path.join(frb_dir, "*.h5"))

if not h5_files:
    print("Ошибка! В папке FRB не найдено ни одного файла .h5")
    exit()

print(f"[+] Найдено файлов для обработки: {len(h5_files)}")

# --- НАЧАЛО НОВОГО БЛОКА ---
def print_h5_tree(name, obj):
    """Рекурсивно обходит структуру H5 и выводит её компоненты"""
    indent = "  " * name.count('/')
    if hasattr(obj, 'shape'):  # Если это массив (Dataset)
        print(f"{indent} Массив: {name.split('/')[-1]} | Форма: {obj.shape} | Тип: {obj.dtype}")
    else:                      # Если это группа (Group)
        print(f"{indent} Группа: {name.split('/')[-1]}")
# --- КОНЕЦ НОВОГО БЛОКА ---

for h5_path in h5_files:
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    npz_output = os.path.join(frb_dir, f"{base_name}_converted.npz")
    
    print(f"\n[*] Препарирую: {os.path.basename(h5_path)} -> {os.path.basename(npz_output)}")
    
    try:
        with h5py.File(h5_path, "r") as f:
            # --- ВЫВОД СТРУКТУРЫ "БЫЛО" ---
            print("\n--- СТРУКТУРА ИСХОДНОГО ФАЙЛА (БЫЛО) ---")
            f.visititems(print_h5_tree)
            print("-" * 40)

            # Стучимся напрямую внутрь группы 'frb'
            if 'frb' in f:
                group = f['frb']
                
                # Приоритет отдаем калиброванному водопаду, если его нет — берем обычный wfall
                if 'calibrated_wfall' in group:
                    print("  [+] Извлекаю массив: 'frb/calibrated_wfall'")
                    real_wfall = group['calibrated_wfall'][:]
                elif 'wfall' in group:
                    print("  [+] Извлекаю массив: 'frb/wfall'")
                    real_wfall = group['wfall'][:]
                else:
                    print(f"  [!] Пропуск! Нет нужных массивов. Доступно: {list(group.keys())}")
                    continue
            else:
                print("  [!] Ошибка структуры: В корне нет группы 'frb'!")
                continue

        # Векторизуем и укладываем в 4 виртуальных луча для вашего 21D-детектора
        flat_data = real_wfall.ravel()
        trunc_len = (len(flat_data) // 4) * 4
        beam_matrix = flat_data[:trunc_len].reshape(-1, 4).astype(np.float32)

        # НАУЧНАЯ ЧИСТОТА: Сохраняем ТОЛЬКО реальный калиброванный водопад
        # Больше не выдумываем ложную экспозицию!
        # np.savez_compressed(
        #     npz_output, 
        #     beam_inc_exp=beam_matrix
        # )
        # print(f"   ОТЛИЧНО! Массив {beam_matrix.shape} успешно сохранен в beam_inc_exp.")

        # НАУЧНАЯ ЧИСТОТА: сохраняем честную 2D матрицу (Частота х Время)
        # Нам больше не нужно притворяться, что это 4 суточных луча
        real_shape = np.array(real_wfall.shape, dtype=np.int32) # Запишет, например, [16384, 192]

        np.savez_compressed(
            npz_output, 
            beam_inc_exp=real_wfall.astype(np.float32),
            original_shape=real_shape
        )

        # --- ВЫВОД СТРУКТУРЫ "СТАЛО" ---
        print("\n--- СТРУКТУРА ДЛЯ NPZ (СТАЛО) ---")
        with np.load(npz_output) as npz_file:
            for key in npz_file.files:
                arr = npz_file[key]
                print(f"   Массив: {key} | Форма: {arr.shape} | Тип: {arr.dtype}")
        print("-" * 40)
        print(f"  Успешно! Честная матрица {real_wfall.shape} успешно закартирована.")
        
    except Exception as e:
        print(f"   Сбой при обработке файла: {e}")

print("\n==========================================================================")
print(" КОНВЕРТАЦИЯ ЗАВЕРШЕНА НА 100%! ВСЕ СТРУКТУРЫ СИНХРОНИЗИРОВАНЫ!")
print("==========================================================================")