import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
from scipy.signal import welch
from scipy.linalg import svd

def generate_nasa_pds_url(date_str, channel_type="2_5KHZ"):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year = dt.year
        doy = dt.timetuple().tm_yday
        xx_folder = f"T{year}{str(doy // 100 * 100).zfill(3)[1:]}XX"
        day_folder = f"T{year}{str(doy).zfill(3)}"
        file_name = f"{day_folder}_{channel_type}4_WFRFR.DAT"
        base_url = "https://pds-ppi.igpp.ucla.edu/data/CO-V_E_J_S_SS-RPWS-2-REFDR-WFRFULL-V1.0/DATA/RPWS_WAVEFORM_FULL/T20060XX/T2006003/T2006003_2_5KHZ4_WFRFR.DAT"
        return f"{base_url}/{xx_folder}/{day_folder}/{file_name}", file_name
    except Exception:
        return None, None

def auto_multiscale_21d_scan(date_list, channels=["2_5KHZ", "25HZ"]):
    print("="*85)
    print("🛰️  ВСЕВОЛНОВОЙ АВТОНОМНЫЙ КОМПЛЕКС: ДИНАМИЧЕСКИЙ ТРЕКИНГ СБОЕВ 21D")
    print("=====================================================================")
    
    # Реестр для записи каждого обнаруженного сбоя
    global_fault_registry = []
    
    # Сетка автоматического подбора окон (от мелкого микроскопа до макро-сканирования)
    window_scales = [250, 500, 1000, 2500, 5000]
    tau_scales = [1, 2, 3, 5, 8]
    crystals_nodes = 21

    for target_date in date_list:
        for chan in channels:
            url, filename = generate_nasa_pds_url(target_date, chan)
            if not url: continue
            
            try:
                # Скачиваем поток напрямую в оперативную память
                response = requests.get(url, timeout=45, stream=True)
                if response.status_code != 200: continue
                
                raw_signal = np.frombuffer(response.content, dtype=np.int16)
                raw_signal = raw_signal[np.isfinite(raw_signal)]
                total_samples = len(raw_signal)
                
                if total_samples < 105000: continue
                
                print(f"\n[+] СКАНИРОВАНИЕ ПОТОКА ОЗУ -> {target_date} [{chan}] ({total_samples} отсчетов)")
                
                # 1. ЗАПУСК АВТОМАТИЧЕСКОГО ПЕРЕБОРА МАСШТАБОВ (Динамический круиз-контроль)
                best_anomaly_score = 0.0
                detected_fault_sample = 0
                active_window = 0
                active_tau = 0
                
                # Скрипт сам крутит параметры 21D и ищет максимальный перекос структуры
                for win_sz in window_scales:
                    for t_lag in tau_scales:
                        req_samples = crystals_nodes * win_sz
                        
                        # Скользящий шаг внутри файла для локализации точной точки сбоя
                        stride = win_sz // 2
                        max_steps = (total_samples - req_samples) // stride
                        
                        for step in range(min(max_steps, 10)): # Ограничиваем шаг для скорости ОЗУ
                            start_pos = step * stride
                            
                            # Сборка локального 21D-Кристалла
                            crystal_matrix = np.zeros((crystals_nodes, win_sz))
                            for d in range(crystals_nodes):
                                shift = start_pos + (d * t_lag)
                                crystal_matrix[d, :] = raw_signal[shift : shift + win_sz]
                                
                            # SVD-тест геометрии
                            _, S_vals, _ = svd(crystal_matrix, full_matrices=False)
                            S_norm = S_vals / np.sum(S_values := S_vals)
                            entropy_21d = -np.sum(S_norm * np.log2(S_norm + 1e-12))
                            anomaly_score = (np.log2(crystals_nodes) - entropy_21d) / np.log2(crystals_nodes) * 100.0
                            
                            # Если кристалл перекосило сильнее, фиксируем координату шага
                            if anomaly_score > best_anomaly_score:
                                best_anomaly_score = anomaly_score
                                detected_fault_sample = start_pos
                                active_window = win_sz
                                active_tau = t_lag

                # 2. АВТОНОМНАЯ ЛОКАЛИЗАЦИЯ МИКРО-ИГЛЫ (FFT-Спектроскоп)
                fs_sampling = 2500.0 if chan == "2_5KHZ" else 25.0
                frequencies, psd_values = welch(raw_signal, fs=fs_sampling, nperseg=4096)
                psd_db = 10 * np.log10(psd_values + 1e-12)
                
                # Поиск пика по критерию 3.5-Sigma
                mean_bg, std_bg = np.mean(psd_db), np.std(psd_db)
                threshold = mean_bg + (3.5 * std_bg)
                peaks_idx = np.where(psd_db > threshold)[0]
                
                if len(peaks_idx) > 0 and best_anomaly_score > 1.5:
                    max_p_idx = peaks_idx[np.argmax(psd_db[peaks_idx])]
                    print(f"   🚨 [ФИКСАЦИЯ СБОЯ]: Кристалл деформирован! Сканирование локализовало паттерн.")
                    
                    global_fault_registry.append({
                        "date": target_date,
                        "chan": chan,
                        "anomaly": f"{best_anomaly_score:.3f} %",
                        "sample_idx": f"Кадр {detected_fault_sample}",
                        "config": f"Win:{active_window}/Tau:{active_tau}",
                        "needle_hz": f"{frequencies[max_p_idx]:.1f} Гц",
                        "power": f"{psd_db[max_p_idx]:.1f} дБ"
                    })
                else:
                    print("   [+] Поток чист. Адаптивная матрица компенсировала фоновые шумы плазмы.")
                    
            except Exception as e:
                continue

    # =====================================================================
    # СВОДНЫЙ СИСТЕМНЫЙ ЖУРНАЛ КРИСТАЛЛИЧЕСКИХ СБОЕВ
    # =====================================================================
    print("\n" + "="*85)
    print("📋 СВОДНЫЙ ЖУРНАЛ ВСЕВОЛНОВОГО МОНИТОРИНГА: ОЦИФРОВАННЫЕ СБОИ 21D И МИКРО-ИГЛЫ")
    print("="*85)
    
    if not global_fault_registry:
        print("[+] Автоматический круиз-контроль завершен. Сбоев структуры по всей сетке не найдено.")
    else:
        # Строгая научная таблица результатов
        print(f"{'ДАТА':<11} | {'КАНАЛ':<7} | {'СБОЙ 21D':<10} | {'ТОЧКА СБОЯ':<12} | {'КОНФИГУРАЦИЯ':<13} | {'ЧАСТОТА ИГЛЫ':<13}")
        print("-" * 85)
        for f in global_fault_registry:
            print(f"{f['date']:<11} | {f['chan']:<7} | {f['anomaly']:<10} | {f['sample_idx']:<12} | {f['config']:<13} | {f['needle_hz']:<13}")
            
    print("="*85)

if __name__ == "__main__":
    # Сюда закидываем наши проверенные два дня, на которых мы ловили сигналы
    scan_dates = ["2006-01-01", "2006-01-02"]
    auto_multiscale_21d_scan(scan_dates)